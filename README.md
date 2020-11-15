from bs4 import BeautifulSoup
import urllib2
url = urllib2.urlopen('http://www.website_address.com')
soup = BeautifulSoup(url)
images = soup.find_all('img')


from bs4 import BeautifulSoup
import urllib2
html = '''
<img src="smiley.gif" alt="Smiley face" height="42" width="42">'''
soup = BeautifulSoup(html)
images = soup.find('img')
print(images['src']) #smiley.gif