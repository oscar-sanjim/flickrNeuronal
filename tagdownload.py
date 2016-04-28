import flickr
from flickr import Photoset
import urllib, urlparse
import os
import sys

if len(sys.argv)>1:
    tag = sys.argv[1]
else:
    print ('no tag specified')

# downloading image data
f = flickr.photos_search(tags="city, day", sort="interestingness-desc", tag_mode="all")
#obj = Photoset(id=72157654804104859)
#f = obj.getPopular()
urllist = [] #store a list of what was downloaded

# downloading images
index = 142
for k in f:
    fileName = "Photo_" + str(index)
    index += 1
    path = "/home/oscar/Documents/ISC/8vo/Sistemas_Inteligentes/Flickr/day"
    url = k.getURL(size='Medium', urlType='source')
    count_photo = k.getFavoriteCount()
    print (count_photo)
    urllist.append(url) 
    image = urllib.URLopener()
    image.retrieve(url, os.path.join(path, fileName))
    print ('downloading:', url)

