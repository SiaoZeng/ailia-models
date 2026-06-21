import sys
import os
import importlib.util

# モデルファイルや相対パスが whisper/ を基準にしているため CWD を変更する
_whisper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "whisper")
os.chdir(_whisper_dir)

# whisper.py がモジュールレベルで引数をパースするため、
# ロード前に --lite_whisper を sys.argv に挿入する
if "--lite_whisper" not in sys.argv:
    sys.argv.insert(1, "--lite_whisper")

# ../whisper/whisper.py をモジュールとしてロード
_whisper_py = os.path.join(_whisper_dir, "whisper.py")
_spec = importlib.util.spec_from_file_location("whisper_main", _whisper_py)
_whisper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_whisper)

if __name__ == "__main__":
    _whisper.main()
