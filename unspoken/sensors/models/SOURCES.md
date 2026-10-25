# Vendored and downloadable sensor models

Small models ship in the repo so the box needs no download on the day. The speaker-embedding model is
larger and is fetched once with `python -m unspoken.sensors.fetch_models` (sha256-checked) into
`$UNSPOKEN_MODELS_DIR` (default `~/.cache/unspoken/models`).

| File | Source (pinned) | Licence | sha256 |
|---|---|---|---|
| `silero_vad.onnx` (2.3 MB) | https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx | MIT (`LICENSE.silero-vad.txt`) | `2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f` |
| `face_detection_yunet_2023mar.onnx` (232 KB) | https://github.com/opencv/opencv_zoo/blob/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx | MIT (`LICENSE.yunet.txt`) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `wespeaker_en_voxceleb_CAM++.onnx` (29 MB, downloaded) | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx (WeSpeaker VoxCeleb CAM++ export, https://github.com/wenet-e2e/wespeaker) | Apache-2.0 (`LICENSE.wespeaker.txt`) | `c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef` |
| `wespeaker_en_voxceleb_resnet34_LM.onnx` (27 MB, optional) | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx | Apache-2.0 | `e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012` |
