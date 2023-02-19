"""WebRTC server that runs HomeVision solution"""
# https://github.com/aiortc/aiortc/tree/main/examples/server
from __future__ import annotations

import asyncio
import concurrent
import functools
import json
import logging
import os
import time
from typing import Optional, Tuple

import aiohttp
import aiohttp_cors
import aiohttp_jinja2
import jinja2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame
from home_vision.modules.module_base import BaseConfig, Module
from home_vision.solutions.solution_base import Solution, SolutionConfig
from home_vision.utils.utils import load_solution

ROOT = os.path.dirname(__file__)

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that process media track using HomeVision solution
    and re-stream the processed frames and outputs
    """

    kind = "video"

    def __init__(self, track: MediaStreamTrack, track_type: str, solution: Solution):
        """Initialize the VideoTransformTrack that re-stream the HomeVision processed video

        Args:
            track (MediaStreamTrack): input media track
            track_type (str): type of track, could be `video`, `stream`, `rtc`
            solution (Solution): HomeVision Solution
        """
        super().__init__()
        self.track = track
        self.track_type = track_type
        self.frame_cnt = 0
        self.fps = 0
        self.channels = set()
        self.solution = solution
        self.pool = concurrent.futures.ThreadPoolExecutor()

    async def recv(self) -> VideoFrame:
        """Run HomeVision solution process and re-stream the processed outputs"""
        loop = asyncio.get_event_loop()
        frame_s = time.perf_counter()

        # gather solution inputs
        kwargs = {}
        if self.track_type == "rtc":
            kwargs, frame = await self.track.recv()
        else:
            frame = await self.track.recv()
            kwargs["image"] = frame.to_ndarray(format="bgr24")
        self.frame_cnt += 1
        frame_e = time.perf_counter()
        # res.json(exclude={'image'})
        solution_input = self.solution.input_types(**kwargs)
        # solution process inputs
        process_s = time.perf_counter()
        res = await loop.run_in_executor(
            self.pool, functools.partial(self.solution.process, solution_input)
        )
        res_image = res.image
        res = res.dict(exclude={'image'})
        if self.track_type != "rtc":
            self.solution.draw_note(res_image, self.fps, self.frame_cnt)
            res['frame_cnt'] = self.frame_cnt
        process_e = time.perf_counter()

        # send processed outputs to peers' datachannels
        if len(self.channels) != 0:
            for channel in self.channels:
                channel.send(
                    json.dumps(res)
                )
        channel_e = time.perf_counter()

        # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(res_image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        frame_e = time.perf_counter()
        self.fps = 1 / (frame_e - frame_s)
        logging.info(
            ("PID: %s | FPS: %s | capturing: %.2f ms | channel: %.2f ms | processing: %.2f ms"
            "| total: %.2f ms") ,os.getpid(), int(self.fps), (process_s - frame_s) * 1000,
            (channel_e - process_e) * 1000, (process_e - process_s) * 1000,
            (frame_e - frame_s) * 1000
        )
        return new_frame


class RTCListener:
    """A `MediaTrack` like RTC peer that connects to a webrtc server that is running HomeVision
    solution, it will receive the frame from track and processed results from datachannel"""
    def __init__(self, url: str):
        """Initialize the RTC listener
        Args:
            url (str): url of the webrtc server that HomeVision solution is running
        """
        self.url = url
        self.pc = None #pylint: disable=invalid-name
        self.data = None
        self.track = None

    async def run_offer(self, url: str):
        """Connect to RTC server

        Args:
            url (str): url of the webrtc server that HomeVision solution is running
        """
        formatted_url = url.replace('"', '')
        self.pc = RTCPeerConnection()
        self.pc.addTransceiver('video', 'recvonly')
        channel = self.pc.createDataChannel("chat")

        @self.pc.on("track")
        def on_track(track):
            """Receive processed media track from connected solution"""
            self.track = track

        @self.pc.on("iceconnectionstatechange")
        def on_iceconnectionstatechange():
            """Peer connection state change"""
            logging.info("RTC listener connection state %s", self.pc.iceConnectionState)

        @channel.on("open")
        async def on_open():
            """Peer connection datachannel open"""
            logging.info("RTC listener datachannel opened!")

        @channel.on("message")
        async def on_message(message):
            """Receive processed results from connected solution's datachannel"""
            self.data = json.loads(message)

        # webrtc connection
        await self.pc.setLocalDescription(await self.pc.createOffer())
        offer = self.pc.localDescription
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{formatted_url}/offer', json={"sdp":offer.sdp, "type": offer.type, "track": True}
            ) as resp:
                answer = await resp.json()

        await self.pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=answer["sdp"], type=answer["type"]
            )
        )


    async def recv(self) -> Tuple[dict, VideoFrame]:
        """re-stream the processed results from connected HomeVision Solution"""
        # start peer connection if not already
        if self.pc is None:
            await self.run_offer(self.url)
            await asyncio.sleep(5)
        # re-stream processed result from datachannel
        if self.data:
            outputs = self.data
        else:
            outputs = {}
        # re-stream the frame
        frame = await self.track.recv()
        outputs['image'] = frame.to_ndarray(format="bgr24")
        return outputs, frame


class RTCConfig(BaseConfig):
    """Config for RTC server

    Attributes:
        source (str): input video/camera source or url which runs WebRTC server
        host (str): hostname that RTC server will run
        port (int): port number that RTC server will run
        buffered (bool): whether to streaming the buffered video
        codec (bool): whether to streaming the compressed video
    """
    source: str
    host: Optional[str] = "0.0.0.0"
    port: Optional[int] = 5555
    buffered: Optional[bool] = False
    codec: Optional[bool] = False

@Module.register('rtc_server')
class RTCServer(Module):
    """WebRTC server that starts HomeVision solution and handles peer connection"""
    config_type = RTCConfig
    module_name = "webrtc_server"

    def __init__(self, source: str, host: str, port: int, buffered: bool, codec: bool):
        self.host = host
        self.port = port
        self.source = source
        self.solution_name = None
        self.solution_config = None
        self.buffered = buffered
        self.codec = codec
        self.pcs = set()
        self.player = None
        self.video = None
        self.relay = None
        self.track_type = None
        self.media_track = None
        self.recorder = MediaBlackhole()
        self.solution = None


    @classmethod
    def from_config(cls, config: RTCConfig) -> RTCServer:
        logging.info('loading RTC server from config: %s', config)
        return cls(config.source, config.host, config.port, config.buffered, config.codec)

    def create_tracks(self) -> MediaStreamTrack:
        """Create the track that re-stream HomeVision Solution results"""
        if self.track_type is None:
            # video streams
            if self.source.startswith('rtsp://') or self.source.startswith('rtmp://'):
                self.track_type = 'stream'
                self.player = MediaPlayer(self.source)
                self.media_track = MediaRelay().subscribe(self.player.video, buffered=self.buffered)
            # url that runs HomeVision Solution WebRTC server
            elif self.source.startswith('http'):
                self.track_type = 'rtc'
                self.player = None
                self.media_track = RTCListener(self.source)
            # local video files
            else:
                self.track_type = 'video'
                self.player = MediaPlayer(self.source, loop=True)
                self.media_track = MediaRelay().subscribe(self.player.video, buffered=self.buffered)
        # load HomeVision Solution
        if self.solution is None:
            self.solution = load_solution(self.solution_name, self.solution_config)
        # create the transform track that process input frames using HomeVision Solution
        if self.video is None:
            self.video = VideoTransformTrack(self.media_track, self.track_type, self.solution)
        # relay for output stream
        if self.relay is None:
            self.relay = MediaRelay()
        return self.relay.subscribe(self.video, False)

    def force_codec(self, pc: RTCPeerConnection, sender: RTCRtpSender, forced_codec: str): #pylint: disable=invalid-name
        """Compress send MediaTrack

        Args:
            pc (RTCPeerConnection): Peer connection that need transfer compressed stream
            sender (RTCRtpSender): Sender of MediaTrack
            forced_codec (str): MediaStream Type. e.g. "video/H264"
        """
        kind = forced_codec.split("/")[0]
        codecs = RTCRtpSender.getCapabilities(kind).codecs
        transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
        transceiver.setCodecPreferences(
            [codec for codec in codecs if codec.mimeType == forced_codec]
        )

    async def index(self, request):
        """Render RTC server's home page"""
        context = {
            'solution_name': self.solution_name,
            'connected_peers': len(self.pcs),
            'connected_channels': len(self.video.channels) if self.video else 0
            }
        response = aiohttp_jinja2.render_template("index.html",
                                                request,
                                                context)
        response.headers['Content-Language'] = 'ru'
        return response

    async def javascript(self, request): #pylint: disable=unused-argument
        """Add javascript RTC client resource"""
        with open(os.path.join(ROOT, "client.js"), "r", encoding="utf-8") as js_file:
            content = js_file.read()
        return web.Response(content_type="application/javascript", text=content)

    async def offer(self, request):
        """Handle Peer Connection request"""
        # make offer based on request
        params = await request.json()
        use_track = params['track']
        logging.info('Request for track: %s', use_track)
        pc = RTCPeerConnection() #pylint: disable=invalid-name
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        self.pcs.add(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            """RTC Peer connection change"""
            logging.info("RTC server ICE connection state is %s", pc.iceConnectionState)
            # clean up when connection ends
            if pc.iceConnectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)
                if len(self.pcs) == 0:
                    if self.video is not None:
                        self.video.stop()
                    if self.player is not None:
                        self.player.video.stop()

                    self.video = None
                    self.player = None
                    self.relay = None
                    self.track_type = None
                    self.recorder.stop()
                    self.recorder = MediaBlackhole()
        # get HomeVision solution re-stream track
        video = self.create_tracks()

        # mediablackhole recorder use to start streaming
        self.recorder.addTrack(video)

        # send transformed video track
        if use_track:
            video_sender = pc.addTrack(video)

        @pc.on("datachannel")
        def on_datachannel(channel):
            """Handle client's requested datachannel"""
            logging.info("RTC server received a datachannel!")
            # add datachannels to HomeVision solution track to send results
            self.video.channels.add(channel)
            @channel.on("close")
            def on_close():
                self.video.channels.remove(channel)

        # compress stream
        if self.codec and use_track:
            self.force_codec(pc, video_sender, "video/H264")

        # complete peer connection
        await pc.setRemoteDescription(offer)
        await self.recorder.start()

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )

    async def on_shutdown(self, app): #pylint: disable=unused-argument
        """clean up and close all peer connections"""
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()
        if self.video is not None:
            self.video.pool.shutdown()
            self.video.stop()
        if self.player is not None:
            self.player.video.stop()
        if self.recorder is not None:
            await self.recorder.stop()


    def _process(self, **kwargs):
        pass

    def create_server(self, solution_name: str, solution_config: SolutionConfig) -> web.Application:
        """Create a WebRTC server that can run HomeVision Solution and re-stream the results

        Args:
            solution_name (str): HomeVision solution name
            solution_config (SolutionConfig): HomeVision solution config

        Returns:
            web.Application: web application of the RTC server
        """
        self.solution_name = solution_name
        self.solution_config = solution_config
        app = web.Application()
        cors = aiohttp_cors.setup(app)
        # setup jinja2
        aiohttp_jinja2.setup(app,
            loader=jinja2.FileSystemLoader(
                os.path.join(ROOT, 'templates')
        ))
        app.on_shutdown.append(self.on_shutdown)
        app.router.add_get("/", self.index)
        app.router.add_get("/client.js", self.javascript)
        resource = cors.add(app.router.add_resource("/offer"))
        cors.add(resource.add_route("POST", self.offer),
            {"*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                max_age=3600,
            )}
        )
        return app

    def run_server(self, solution_name: str, solution_config: SolutionConfig):
        """Start a WebRTC server that runs HomeVision solution and re-stream the results

        Args:
            solution_name (str): HomeVision solution name
            solution_config (SolutionConfig): HomeVision solution config
        """
        app = self.create_server(solution_name, solution_config)
        web.run_app(app, host=self.host, port=self.port)
