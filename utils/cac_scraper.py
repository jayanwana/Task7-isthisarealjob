import time
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
import itertools
import argparse


class WebTable:
    def __init__(self, webtable):
        self.table = webtable

    def get_row_count(self):
        return len(self.table.find_elements_by_tag_name("tr")) - 1

    def get_column_count(self):
        return len(self.table.find_elements_by_xpath("//tr[2]/td"))

    def get_table_size(self):
        return {"rows": self.get_row_count(),
                "columns": self.get_column_count()}

    def row_data(self, row_number):
        if row_number == 0:
            raise Exception("Row number starts from 1")

        row_number = row_number + 1
        row = self.table.find_elements_by_xpath("//tr["+str(row_number)+"]/td")
        r_data = []
        for webElement in row:
            r_data.append(webElement.text)

        return r_data

    def column_data(self, column_number):
        col = self.table.find_elements_by_xpath("//tr/td["+str(column_number)+"]")
        r_data = []
        for webElement in col:
            r_data.append(webElement.text)
        return r_data

    def get_all_data(self):
        # get number of rows
        no_of_rows = len(self.table.find_elements_by_xpath("//tr")) - 1
        # get number of columns
        no_of_columns = len(self.table.find_elements_by_xpath("//tr[2]/td"))
        all_data = []
        # iterate over the rows, to ignore the headers we have started the i with '1'
        for i in range(2, no_of_rows):
            # reset the row data every time
            ro = []
            # iterate over columns
            for j in range(1, no_of_columns):
                # get text from the i th row and j th column
                ro.append(self.table.find_element_by_xpath("//tr["+str(i)+"]/td["+str(j)+"]").text)

            # add the row data to all_data of the self.table

            all_data.append(ro)

        return all_data

    def presence_of_data(self, data):

        # verify the data by getting the size of the element matches based on the text/data passed
        data_size = len(self.table.find_elements_by_xpath("//td[normalize-space(text())='"+data+"']"))
        presence = False
        if data_size > 0:
            presence = True
        return presence

    def get_cell_data(self, row_number, column_number):
        if row_number == 0:
            raise Exception("Row number starts from 1")

        row_number += 1
        cell_data = self.table.find_element_by_xpath("//tr["+str(row_number)+"]/td["+str(column_number)+"]").text
        return cell_data


def get_string(length=3, characters=r'abcdefghijklmnopqrstuvwxyz '):
    for s in itertools.product(characters, repeat=length):
        yield ''.join(s)


def join_db(db):
    cac_db = pd.read_csv(r'data\cac_db.csv')
    cac_db_main = pd.concat([cac_db, db])
    cac_db_main = cac_db_main.drop_duplicates()
    cac_db_main.sort_values(by='COMPANY NAME', inplace=True)
    cac_db_main.reset_index(drop=True, inplace=True)
    cac_db_main.to_csv(r'data\cac_db.csv', index=None, header=True)


def cac_scraper(x, options=None):
    """Scrapes CAC website for company info
    args:
    x: str, letter combination to continue scraping from
    """

    chrome_path = r"data\chromedriver.exe"
    string = [s for s in get_string()]
    try:
        idx = string.index(x)
    except ValueError:
        return "Please enter a 3 letter alphabet string "
    try:
        if options is not None:
            print('using chrome user data')
            options_path = r"--user-data-dir=" + options
            options = webdriver.ChromeOptions()
            options.add_argument(options_path)
            driver = webdriver.Chrome(executable_path=chrome_path, chrome_options=options)
        else:
            driver = webdriver.Chrome(executable_path=chrome_path)

    except:
        return "Problem loading Chrome driver, check file path"
    url = 'http://publicsearch.cac.gov.ng/comsearch/'
    driver.get(url)
    time.sleep(10)
    data = []
    scraped = []
    for s in string[idx:]:
        try:
            driver.find_element_by_css_selector('input.field').send_keys(s)
            WebDriverWait(driver, 300).until(ec.presence_of_element_located((By.ID, "directorsTbl")))

            table = driver.find_element_by_id('directorsTbl')

            cac_table = WebTable(table)
            if cac_table.get_row_count() > 1:
                db = []
                print(f'collecting {s} data')
                for i in range(cac_table.get_row_count()):
                    row = i + 1
                    row_data = cac_table.row_data(row)
                    db.append(row_data[:-1])

                data.append(db)
                scraped.append(s)
            else:
                pass
            driver.refresh()
        except:
            print(f"database creation stopped at {s}")
            break
    cac_data = []
    for _ in data:
        for d in _:
            cac_data.append(d)
    col_names = ['RC NUMBER', 'COMPANY NAME', 'ADDRESS']
    cac_db = pd.DataFrame(cac_data, columns=col_names)
    join_db(cac_db)
    return f'Data for {scraped} has been collected'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('search_string', type=str,
                        help='Enter a 3 letter string you wish to start scraping from')
    parser.add_argument('--options', type=str, default=r'C:\Users\user\AppData\Local\Google\Chrome\User Data\Default',
                        help='Path to Google Chrome user data for captcha ease')
    args = parser.parse_args()
    # options = r'C:\Users\user\AppData\Local\Google\Chrome\User Data\Default'
    print(cac_scraper(args.search_string))


if __name__ == '__main__':
    main()
