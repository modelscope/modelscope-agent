from PIL import Image
import base64
import io

# 图片本地路径转换为 base64 格式
def covert_image_to_base64(image_path):
  image = Image.open(image_path).convert("RGB")
  # 创建一个内存字节流
  image_stream = io.BytesIO()
  # 将图片保存到字节流中，格式自动识别
  image.save(image_stream, format="JPEG")
  # 获取字节流内容
  image_data = image_stream.getvalue()
  # 转换为base64编码
  base64_data = base64.b64encode(image_data).decode('utf-8')
  # 生成base64编码的地址
  base64_url = f"data:image/jpeg;base64,{base64_data}"
  return base64_url

def format_cover_html(configuration, bot_avatar_path):
  if bot_avatar_path:
    image_src = covert_image_to_base64(bot_avatar_path)
  else:
    image_src = "//img.alicdn.com/imgextra/i3/O1CN01YPqZFO1YNZerQfSBk_!!6000000003047-0-tps-225-225.jpg"
  return f"""
<div class="bot_cover">
    <div class="bot_avatar">
        <img src={image_src} />
    </div>
    <div class="bot_name">{configuration.get("name", "")}</div>
    <div class="bot_desp">{configuration.get("description", "")}</div>
</div>
"""