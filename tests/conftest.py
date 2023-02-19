# """global fixtures for test"""
import asyncio
import json
from typing import Callable, List

import pytest_asyncio
from aiohttp import web
from aiortc import RTCDataChannel, RTCPeerConnection
from home_vision.modules.module_base import Module
from home_vision.modules.rtc_server.server import RTCServer
from home_vision.solutions.solution_base import Solution, SolutionConfig

# pylint: disable=invalid-name

def rtc_server(
    solution_name: str,
    video_src: str,
    solution_config: SolutionConfig=None
) -> web.Application:
    """Load a HomeVision webrtc server"""
    rtc_dict = {
        'source': video_src,
        'port': 8080,
        'codec': False,
        'buffered': True
    }
    module_cls: RTCServer = Module.by_name('rtc_server')
    rtc_config = module_cls.config_type(**rtc_dict)
    rtc_server_cls = module_cls.from_config(rtc_config)
    if solution_config is None:
        solution_config = Solution.by_name(solution_name).config_type()
    server = rtc_server_cls.create_server(solution_name, solution_config)
    return server

@pytest_asyncio.fixture
async def rtc_client(aiohttp_client, request):
    """Load rtc aiohttp client based on solution name"""
    server = rtc_server(request.param, "tests/test.mp4")
    client = await aiohttp_client(server)
    return client

async def sleepWhile(f: Callable, max_sleep: float=15.0):
    """Async sleep till function returns `True`"""
    sleep = 1
    total = 0.0
    while f() and total < max_sleep:
        await asyncio.sleep(sleep)
        total += sleep

async def assertDataChannelOpen(dc: RTCDataChannel):
    """Check datachannel is opened"""
    await sleepWhile(lambda: dc.readyState == "connecting")
    assert dc.readyState == "open"

async def assertEnoughMessages(messages: List, total_message: int=100):
    """Check if received enough messages from datachannel"""
    await sleepWhile(lambda: len(messages) < total_message, 35)
    assert len(messages) >= total_message

async def assertIceChecking(pc: RTCPeerConnection):
    """Check peer connection state is checking"""
    await sleepWhile(lambda: pc.iceConnectionState == "new")
    assert pc.iceConnectionState == "checking"
    assert pc.iceGatheringState == "complete"

async def assertIceCompleted(pc: RTCPeerConnection):
    """Check peer connection state is completed"""
    await sleepWhile(
        lambda: pc.iceConnectionState in ("checking", "new")
    )
    assert pc.iceConnectionState == "completed"

def track_remote_tracks(pc: RTCPeerConnection) -> List:
    """Record all received mediatrack from other peer to list"""
    tracks = []

    @pc.on("track")
    def track(track):
        tracks.append(track.kind)

    return tracks

def track_states(pc: RTCPeerConnection):
    """Track RTC peer connection states"""
    states = {
        "connectionState": [pc.connectionState],
        "iceConnectionState": [pc.iceConnectionState],
        "iceGatheringState": [pc.iceGatheringState],
        "signalingState": [pc.signalingState],
    }

    @pc.on("connectionstatechange")
    def connectionstatechange():
        states["connectionState"].append(pc.connectionState)

    @pc.on("iceconnectionstatechange")
    def iceconnectionstatechange():
        states["iceConnectionState"].append(pc.iceConnectionState)

    @pc.on("icegatheringstatechange")
    def icegatheringstatechange():
        states["iceGatheringState"].append(pc.iceGatheringState)

    @pc.on("signalingstatechange")
    def signalingstatechange():
        states["signalingState"].append(pc.signalingState)

    return states

def track_messages(dc: RTCDataChannel) -> List:
    """Record all messages from datachannel to a list"""
    messages = []
    frame_cnts = []

    @dc.on("message")
    async def on_message(message):
        msg_result = json.loads(message)
        if msg_result['frame_cnt'] not in frame_cnts:
            frame_cnts.append(msg_result['frame_cnt'])
            messages.append(message)
    return messages
