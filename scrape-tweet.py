# scrape_tweet.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import pickle
import pandas as pd

# Setup browser
options = Options()
options.add_argument("--headless")  # Sekarang bisa headless karena sudah login
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=options)

# Buka Twitter dan set cookie
driver.get("https://twitter.com/")
time.sleep(3)

# Load cookies
with open("twitter_cookies.pkl", "rb") as f:
    cookies = pickle.load(f)
    for cookie in cookies:
        driver.add_cookie(cookie)

# Refresh halaman setelah cookie ditambahkan
# driver.get("https://twitter.com/search?q=perang%20dagang%20amerika%20dan%20china%20lang%3Aid&src=typed_query&f=live")
driver.get("https://twitter.com/search?q=tarif%20dagang%20amerika%20china%20lang%3Aid&src=typed_query&f=live")
time.sleep(5)

# Mulai scraping
tweets = set()
last_height = driver.execute_script("return document.body.scrollHeight")

while len(tweets) < 500:  # Ganti jadi 5000 jika perlu
    elements = driver.find_elements(By.XPATH, '//div[@data-testid="tweetText"]')
    for el in elements:
        tweets.add(el.text)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

driver.quit()

# Simpan hasil
df = pd.DataFrame(tweets, columns=["text"])
df.to_csv("tweets_selenium2.csv", index=False)
print(f"Selesai. Total tweet: {len(df)}")
