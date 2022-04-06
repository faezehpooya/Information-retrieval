from selenium import webdriver
import time
import json
import os

class Crawler:
    def __init__(self, start_file_name):
        self.start = start_file_name
        self.queue = []
        self.cache = []

        self.init_queue()

    def init_queue(self):
        with open(self.start, 'r') as file:
            urls = file.readlines()
            for url in urls:
                if 'https' in url:
                    self.queue.append(url)
                    self.cache.append(url)

    def crawl(self, url):
        options = webdriver.ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("--headless")
        options.add_argument("user-agent=Fateme")
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options, executable_path="C:/Users/Fatemeh/Downloads/chromedriver.exe")
        driver.get(url)
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # print(driver.page_source)

        # try:
        #     paper_id = driver.find_element_by_css_selector \
        #         ('#mainArea > router-view > div > div > div > div > a.doiLink.au-target') \
        #         .get_attribute('data-appinsights-paper-doi')
        # except:
        #     paper_id = None
        paper_id = url.split('/')[4].split('\n')[0]
        title = driver.find_element_by_css_selector('#mainArea > router-view > div > div > div > div > h1').text
        abstract = driver.find_element_by_css_selector('#mainArea > router-view > div > div > div > div > p').text
        year = driver.find_element_by_css_selector(
            '#mainArea > router-view > div > div > div > div > a.au-target.publication > span.year').text
        authors_raw = driver.find_elements_by_css_selector(
            '#mainArea > router-view > div > div > div > div > ma-author-string-collection > div > div > div > a.au-target.author.link')
        authors = []
        for a in authors_raw:
            authors.append(a.get_attribute('aria-label'))
        atia_ref = driver.find_elements_by_css_selector('div.primary_paper > a.title.au-target')
        references = []
        num_of_links = 0
        for ref in atia_ref:
            ref_link = ref.get_attribute('href')
            if ref_link:
                references.append(ref_link)
                num_of_links += 1
                if not (ref_link in self.cache):
                    self.cache.append(ref_link)
                    self.queue.append(ref_link)
                if num_of_links >= 10:
                    break

        # print(paper_id)
        # print(title)
        # print(abstract)
        # print(year)
        # print(authors)
        # print(references)

        driver.quit()

        return {'id': paper_id, 'title': title, 'abstract': abstract, 'date': year,
                'authors': authors, 'references': references}

    def crawl_all(self, limit=5000, to_store_file='crawler_data/papers', store=False):
        if store and os.path.exists(to_store_file+'.json'):
            os.remove(to_store_file+'.json')
        num_processed_links = 0
        json_results = []
        time_start = time.time()
        while (len(self.queue) > 0) and (num_processed_links < limit):
            url = self.queue.pop(0)
            print(url)
            while True:
                try:
                    json_result = self.crawl(url)
                    time.sleep(0.2)
                    break
                except Exception as e:
                    print(e)
            json_results.append(json_result)
            num_processed_links += 1
            if num_processed_links % 10 == 0:
                print(num_processed_links, 'links, 10 More! Time elapsed: {} seconds'.format(time.time() - time_start))
                time_start = time.time()

        # print(json_results)
                if store:
                    json.dump(json_results, open(to_store_file+str(num_processed_links)+'.json', 'w'))

        json.dump(json_results, open(to_store_file + '.json', 'w'))

        with open('urls.txt', 'w') as f:
            for item in self.cache:
                item = item.split('\n')[0]
                f.write("%s\n" % item)


if __name__ == '__main__':
    crawler = Crawler('crawler_data/start.txt')
    crawler.crawl_all(store=True)