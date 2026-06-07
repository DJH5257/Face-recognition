# Model Weights

This directory is for local model files only. ONNX files are ignored by Git on purpose.

Expected default files:

```text
models/MiniFASNetV1SE.onnx
models/MiniFASNetV2.yakhyo.onnx
```

The demo also uses InsightFace `buffalo_l`, which is downloaded by the `insightface` package into the user's local InsightFace cache on first run.

## License Notes

The source code in this repository is licensed separately from third-party model weights.

- InsightFace source code is MIT licensed, but its auto-downloaded model packs, including `buffalo_l`, are documented by InsightFace as non-commercial research model assets. Review the upstream license before production or commercial use: <https://github.com/deepinsight/insightface/wiki/Dataset-Zoo>.
- MiniFASNet anti-spoofing weights may come from different upstream exports. Common references include Minivision's Silent-Face-Anti-Spoofing project and ONNX conversions such as yakhyo/face-anti-spoofing, but you still need to keep the source URL and upstream license for the exact files you use before redistributing them:
  - <https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>
  - <https://github.com/yakhyo/face-anti-spoofing>

Do not publish biometric samples, enrollment images, camera captures, or private test data in this directory.
