"""Download official lightweight target-speaker model dependencies into local cache."""
from modelscope import snapshot_download

if __name__ == "__main__":
    for model in ("iic/speech_campplus_sv_zh-cn_16k-common",
                  "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"):
        print(snapshot_download(model))
