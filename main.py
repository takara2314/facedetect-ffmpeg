# 参考: https://symfoware.blog.fc2.com/blog-entry-2413.html

import os
import shutil
import sys
import ffmpeg
import cv2
import numpy as np

# 設定ファイル
prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

# 検出した部位の信頼度の下限値
confidence_limit = 0.8
# 設定ファイルからモデルを読み込み
net = cv2.dnn.readNetFromCaffe(prototxt, model)



# video.mp4 が見つからなければ警告
if not os.path.exists("video.mp4"):
	sys.exit("対象にする動画のファイルを置いてください。また名前は \"video.mp4\" にしてください。")
else:
	print("\"video.mp4\" を読み込みます。")


# stillsフォルダ(静止画保存フォルダ)がなければ作成
if not os.path.exists("stills"):
	os.mkdir("stills")
else:
	print("新しく動画を読み込むために、stillsフォルダの中をすべて削除します。")

	if input("OK(y) / NG(other): ") == "y":
		try:
			shutil.rmtree("stills")
		except PermissionError:
			sys.exit("stillsフォルダを閉じてからもう一度お試しください。")
			
		os.mkdir("stills")
		print("フォルダの中身をすべて削除しました。")

	else:
		sys.exit("ファイルの中身を消せる状態にしてから実行してください。")


print("動画を30FPSで読み込み、静止画に変換します。少々お待ちください。")
shutil.copy("video.mp4", "stills")
# 作業ディレクトリをstillsにし、FFmpegでの変換コマンドを実行結果を非表示で実行
os.chdir("stills")
# 入力
stream = ffmpeg.input("video.mp4")
# 出力
stream = ffmpeg.output(stream, "image_%04d.png", vcodec="png", r=30)
# 実行
ffmpeg.run(stream)
os.remove("video.mp4")
print("変換しました。")


print("各静止画から顔を検知します。")
counter = 1
os.mkdir("rectangled")

for pngData in os.listdir("./"):
	if not os.path.isfile(pngData):
		continue

	print("loaded:", pngData)

	# 解析対象の画像を読み込み
	image = cv2.imread(pngData)
	# 600x600に画像をリサイズ、画素値を調整
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (600, 600)), 1.0,
		(600, 600), (104.0, 177.0, 123.0))


	# 顔検出の実行
	net.setInput(blob)
	detections = net.forward()

	# 検出部位をチェック
	for i in range(0, detections.shape[2]):
		# 信頼度
		confidence = detections[0, 0, i, 2]

		# 信頼度が下限値以下なら無視
		if confidence < confidence_limit:
			continue

		# 検出結果を描画
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		text = '{:.2f}%'.format(confidence * 100)

		y = startY - 10 if startY - 10 > 10 else startY + 10

		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	output_file_name = "rectangled\\image_%04d.png" % counter
	cv2.imwrite(output_file_name, image)
	counter += 1

print("検知しました。")


print("1つの動画に変換します。")
os.chdir("rectangled")

# 入力
stream = ffmpeg.input("image_%04d.png")
# 出力
stream = ffmpeg.output(stream, "output.mp4", vcodec="libx264", pix_fmt="yuv420p", r=30)
# 実行
ffmpeg.run(stream)

shutil.move("output.mp4", "../../output.mp4")
print("変換しました。")