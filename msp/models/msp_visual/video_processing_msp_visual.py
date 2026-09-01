import torch
import torchvision.transforms.v2.functional as tvF
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import PILImageResampling
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.video_processing_utils import BaseVideoProcessor, VideoMetadata
from transformers.video_utils import VideoInput


class MSPVisualVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR

    def __init__(
        self,
        do_convert_rgb_to_grayscale: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        image_mean=0.421,
        image_std=0.165,
        do_normalize: bool = True,
        do_resize: bool = True,
        size: dict[str, int] = {"height": 96, "width": 96},
        do_center_crop: bool = True,
        crop_size: dict[str, int] = {"height": 88, "width": 88},
        **kwargs: Unpack[VideosKwargs],
    ):
        super().__init__(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            image_mean=image_mean,
            image_std=image_std,
            do_normalize=do_normalize,
            do_resize=do_resize,
            size=size,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            **kwargs,
        )
        self.do_convert_rgb_to_grayscale = do_convert_rgb_to_grayscale

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ):
        if num_frames:
            total_frames = metadata.total_num_frames
            num_frames = num_frames if num_frames is not None else self.num_frames
            assert num_frames is not None, (
                "`num_frames` must be specified if `fixed_len_video == True`"
            )
            frame_idxs = [
                int(i * (total_frames - 1) / (num_frames - 1))
                for i in range(num_frames)
            ]
            return torch.tensor(frame_idxs)
        else:
            return super().sample_frames(metadata, num_frames, fps, **kwargs)

    def _load_video(self, src: str | bytes) -> torch.Tensor:
        """
        Load video from a file path or bytes and return as a 4D torch.Tensor.
        Args:
            src (str | bytes): Path to the video file or bytes of the video file.
        Returns:
            torch.Tensor: Loaded video as a 4D tensor (num_frames, height, width, num_channels).
        """
        from torchcodec.decoders import VideoDecoder

        vd = VideoDecoder(src)
        video = vd.get_frames_in_range(0, vd.metadata.num_frames).data
        return video

    def __call__(
        self, videos: VideoInput | str | list[str] | bytes | list[bytes], **kwargs
    ):
        """Overrides the __call__ method to handle video input as file paths or bytes."""
        if isinstance(videos, (str, bytes)):
            videos = self._load_video(videos)
        elif isinstance(videos, list) and isinstance(videos[0], (str, bytes)):
            videos = [self._load_video(v) for v in videos]

        return super().__call__(videos, **kwargs)

    def convert_rgb_to_grayscale(self, video: torch.Tensor) -> torch.Tensor:
        """
        Convert a video to grayscale.
        """
        video = tvF.rgb_to_grayscale(video)
        return video

    def _preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[VideosKwargs],
    ) -> BatchFeature:
        """
        Preprocesses a video or a batch of videos.
        Args:
            videos (VideoInput): Video to preprocess.
                See `VideoInput` for details.
            **kwargs: Additional keyword arguments.
        Returns:
            BatchFeature: A BatchFeature with the following fields:
                - pixel_values_videos: Pixel values to be fed to a model, of shape (batch_size,num_channels, num_frames, height, width).
                - padding_mask_videos (optional): Mask to be used for padding, of shape (batch_size, num_frames).
        """

        # Always set `return_tensors` to `None` since it won't pad variable length videos
        # We'll handle this after we call the parent' method
        return_tensors = kwargs.pop("return_tensors", None)
        result = super()._preprocess(videos, **kwargs)
        pixels = result.pixel_values_videos
        if self.do_convert_rgb_to_grayscale:
            pixels = [self.convert_rgb_to_grayscale(video) for video in pixels]
        data = {"pixel_values_videos": pixels}
        if return_tensors:
            lengths = torch.tensor([video.size(0) for video in pixels])
            pixels = torch.nn.utils.rnn.pad_sequence(
                pixels, batch_first=True, padding_value=0.0
            )
            data["pixel_values_videos"] = pixels
            if lengths.unique().size(0) > 1:
                mask = torch.arange(lengths.max())[None] < lengths[:, None]
                data["padding_mask_videos"] = mask
        # pixel_values_videos shape [batch_size, num_channels, num_frames, height, width]
        data["pixel_values_videos"] = data["pixel_values_videos"].permute(0, 2, 1, 3, 4)

        return BatchFeature(data=data, tensor_type=return_tensors)
