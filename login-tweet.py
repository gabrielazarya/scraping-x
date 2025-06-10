# login_twitter.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import pickle

options = Options()
# Jangan pakai headless biar bisa login manual
driver = webdriver.Chrome(options=options)

# Buka halaman login Twitter
driver.get("https://twitter.com/login")
time.sleep(60)  # Waktu untuk login manual (ubah sesuai kebutuhan)

# Setelah login, simpan cookies
with open("twitter_cookies.pkl", "wb") as f:
    pickle.dump(driver.get_cookies(), f)

print("Cookies disimpan.")
driver.quit()
