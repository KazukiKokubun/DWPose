# DWPose 環境設定

以下は、PyPI版のDWPoseの実行環境作成と実行手順の概要です。
（最初はGitHub版で環境を作成していましたが、実行に必要なファイルのリンク先が無効になっていて、コピーを集めることになりそうです。精度はわかりませんが、対応するPythonのバージョンはPyPI版のほうが新しいです。）

---

## 1. ubuntu環境内に仮想環境を作成

* **環境名**: `.dwpose_venv`など
* **場所**: `ubuntu内ならどこでも`
* **Pythonバージョン**: Python 3.9以降
* **想定マシンスペック**: CUDA11.8対応GPUがあること


* **仮想環境を作成**:

  ```bash
  # Python 3.10 の仮想環境を作成
  python3.10 -m venv dwpose_venv
  ```

* **仮想環境を有効化**:

  ```bash
  source .dwpose_venv/bin/activate
  ```


* **ライブラリをインストール**:

* **pipのアップデート(任意)**:

  ```bash
  pip install --upgrade pip
  ```
  
*  **pytorchのインストール**:

    https://pytorch.org/ からCUDA11.8をインストール

    ```bash
    # 例
    pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
    ```

* **DWPoseに必要なライブラリをpipでインストール**:

    ```bash
    pip install dwpose \
                onnxruntime onnx \
                opencv-python scikit-image scipy \
                matplotlib \
                huggingface-hub
  
    # dwpose：本体
    # onnxruntime／onnx：ONNX モデルランタイム
    # opencv-python：画像読み込み・前処理
    # scikit-image／scipy：追加処理
    # matplotlib：可視化
    # huggingface-hub：モデルの自動ダウンロード
    ```


## 2. 簡単な推定を実行

* **ディレクトリの準備**:

    ```bash
    # 推定画像と出力画像を入れるフォルダ
    mkdir -p frames outputs
    ```

* **フレーム画像がない場合、ffmpegで作成**
    ```bash
    # input_video_path.mp4は動画パスに変更
    input_video_path.mp4 frames/frame_%04d.jpg
    ```

* **テストプログラムの作成・保存**:
    ```bash
    # test_dwpose.pyとして次のプログラムを保存
    from PIL import Image
    from dwpose import DwposeDetector

    # 入力画像を読み込み
    img = Image.open("frames/frame_0001.jpg")

    # モデルの読み込み（自動ダウンロード）
    model = DwposeDetector.from_pretrained_default()

    # 推論実行（人物のbodyのみ、顔・手は無視）
    img_out, keypoints, src = model(
        img,
        include_body=True,
        include_face=False,
        include_hand=False,
        image_and_json=True,
        detect_resolution=512
    )

    # 出力画像を保存
    img_out.save("outputs/frame_0001_pose.jpg")
    ```

* **テストプログラムの実行**:

    ```bash
    python test_dwpose.py
    ```

* **framesに黒い背景のスケルトン画像が1枚生成されていれば環境構築完了**



## 3. frameフォルダ内の全画像に対して推定を実行

下のプログラムを実行すると、
* **キーポイントが打たれた画像**
* **骨格の推定座標が保存されたcsv**

    がoutputフォルダに保存される

    ```bash
    import os
    import csv
    from PIL import Image, ImageDraw
    from dwpose import DwposeDetector

    # ===== 設定 =====
    input_dir = "frames"
    output_dir = "outputs_corrected"
    csv_path = os.path.join(output_dir, "keypoints.csv")
    threshold = 0.01
    detect_resolution = 512   # 固定値にします
    os.makedirs(output_dir, exist_ok=True)

    # COCOキーポイント名
    keypoint_names = [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ]

    # モデル初期化
    model = DwposeDetector.from_pretrained_default()

    # 画像一覧
    files = sorted(f for f in os.listdir(input_dir)
                if f.lower().endswith(('.jpg','jpeg','.png')))

    # CSVヘッダー
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        header = ['image','person_id'] + \
            [f"{n}_{axis}" for n in keypoint_names for axis in ("x","y","confidence")]
        writer.writerow(header)

        # バッチ処理
        for idx, fn in enumerate(files, start=1):
            inp = os.path.join(input_dir, fn)
            img = Image.open(inp).convert("RGB")
            orig_w, orig_h = img.size

            # 推論（固定 detect_resolution）
            img_out, data, _ = model(
                img,
                include_body=True,
                include_face=False,
                include_hand=False,
                image_and_json=True,
                detect_resolution=detect_resolution
            )

            # モデル上のキャンバス情報
            canvas_w = data['canvas_width']
            canvas_h = data['canvas_height']

            # 余白量を計算
            # 画像はアスペクト比を保ってリサイズ → パディングは余った方向にのみ入る
            if orig_w / orig_h > 1:
                # 横長元画像 → 高さ512に合わせ、横は縮小 ⇒ 余白左右ゼロ、垂直方向に余白
                new_h = detect_resolution
                new_w = int(orig_w / orig_h * detect_resolution)
                pad_x = 0
                pad_y = (canvas_h - new_h) / 2
            else:
                # 縦長元画像 → 幅512に合わせ、縦は縮小 ⇒ 余白上下ゼロ、水平方向に余白
                new_w = detect_resolution
                new_h = int(orig_h / orig_w * detect_resolution)
                pad_x = (canvas_w - new_w) / 2
                pad_y = 0

            # リサイズ後 → 元画像へのスケール
            scale_x = orig_w / new_w
            scale_y = orig_h / new_h

            people = data.get('people', [])
            if not people:
                print(f"[{idx}/{len(files)}] skip (no person): {fn}")
                continue

            # 元画像をキャンバスに
            canvas = img.copy()
            draw = ImageDraw.Draw(canvas)

            for pid, person in enumerate(people):
                flat = person['pose_keypoints_2d']
                kpts = [(flat[i], flat[i+1], flat[i+2]) for i in range(0, len(flat), 3)]

                # CSV行
                row = [fn, pid]
                for x, y, c in kpts:
                    if c < threshold:
                        row += ["", "", ""]
                    else:
                        # パディング除去 → スケーリング
                        x0 = (x - pad_x) * scale_x
                        y0 = (y - pad_y) * scale_y
                        row += [x0, y0, c]
                writer.writerow(row)

                # 点描画
                for x, y, c in kpts:
                    if c > threshold:
                        x0 = (x - pad_x) * scale_x
                        y0 = (y - pad_y) * scale_y
                        r = 3
                        draw.ellipse((x0-r, y0-r, x0+r, y0+r), fill=(255,0,0))

            # 出力画像保存
            out_fn = fn.rsplit('.',1)[0] + '_pose.jpg'
            out_path = os.path.join(output_dir, out_fn)
            canvas.save(out_path)
            print(f"[{idx}/{len(files)}] Saved: {out_path}")
    ```




# DWPose
