import requests

# 目标地址
url = "http://jzyhse.supco.com:8080/lake/rtdBroker/d97e0b9133/api/v1/value/current"

# 请求头（带上你提供的 authorization 签名）
headers = {
    "authorization": "Sign e248703b144ba5a45688b9de61ccb0a4-019afff416ee873e087aaab9bbea8cff9015badbc2c14b7362ab65c343caa617",
    "group": "dt",
    "Content-Type": "application/json",
}

try:
    # 发送 GET 请求
    response = requests.post(url, headers=headers, timeout=10)

    # 打印状态码和返回内容
    print("状态码:", response.status_code)
    print("返回内容:\n", response.text)

except Exception as e:
    print("请求出错:", e)
