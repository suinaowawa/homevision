"""Test HomeVision WebRTC server"""
import pytest
from aiortc import (RTCConfiguration, RTCIceServer, RTCPeerConnection,
                    RTCSessionDescription)

from tests.conftest import (assertDataChannelOpen, assertIceCompleted,
                            track_remote_tracks, track_states)


@pytest.mark.parametrize('rtc_client', [
    ('raw_stream_solution')
], indirect=True)

async def test_start_rtc_server(rtc_client):
    """Test rtc server module can be started"""
    resp = await rtc_client.get("/")
    assert resp.status == 200
    text = await resp.text()
    assert text is not None

@pytest.mark.parametrize(
    "rtc_client, with_track, remote_tracks",
    [
        ('raw_stream_solution', True, ['video']),
        ('raw_stream_solution', False, [])
    ], indirect=['rtc_client']
)

@pytest.mark.asyncio
async def test_rtc_connection(rtc_client, with_track, remote_tracks):
    """Test can connect a peer to a HomeVision running rtc server"""
    # create peer connection
    config = RTCConfiguration(
        iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
    )
    # pylint: disable=invalid-name
    pc = RTCPeerConnection(configuration=config)
    pc_states = track_states(pc)
    receive_tracks = track_remote_tracks(pc)
    pc.addTransceiver('video', 'recvonly')

    assert pc.iceConnectionState == "new"
    assert pc.iceGatheringState == "new"
    assert pc.localDescription is None
    assert pc.remoteDescription is None

    # create datachannel
    channel = pc.createDataChannel("chat")
    assert channel.readyState == "connecting"

    @channel.on("message")
    async def on_message(message):
        assert isinstance(message, str)

    # send offer
    await pc.setLocalDescription(await pc.createOffer())
    offer = pc.localDescription
    assert offer.type == "offer"
    assert "m=video" in offer.sdp
    assert "m=application" in offer.sdp

    # receive answer
    resp = await rtc_client.post(
        '/offer', json={"sdp": offer.sdp, "type": offer.type, "track": with_track}
    )
    answer = await resp.json()
    assert answer["type"] == "answer"
    assert "m=video" in answer["sdp"]
    assert "m=application" in answer["sdp"]
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )
    # check ice connection complete
    await assertIceCompleted(pc)

    # check datachannel open
    await assertDataChannelOpen(channel)

    # check receive tracks types
    assert receive_tracks == remote_tracks

    # close connection
    await pc.close()
    await rtc_client.close()

    assert pc.iceConnectionState == "closed"
    assert channel.readyState == "closed"

    # check all tracked pc states
    assert pc_states["connectionState"] == ["new", "connecting", "connected", "closed"]
    assert pc_states['iceConnectionState'] == ["new", "checking", "completed", "closed"]
    assert pc_states['iceGatheringState'] == ["new", "gathering", "complete"]
    assert pc_states['signalingState'] == ["stable", "have-local-offer", "stable", "closed"]
