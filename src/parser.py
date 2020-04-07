import bs4
from readability.readability import Document as Paper

TEXT_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']
CONTENT_CLASSES = ['content-media__description']


def element_is_dispensable(soup_element):
    # if 'class' in soup_element.attrs and len(soup_element.attrs['class']) > 0 and set(soup_element.attrs['class']).issubset(CONTENT_CLASSES):
    #     return True
    # if 'itemprop' in soup_element.attrs and soup_element.attrs['itemprop'] == 'description':
    #     return True
    return False


def html_cleaner(html):
    html = Paper(html).summary()
    soup = bs4.BeautifulSoup(html, 'lxml')

    return "\n".join([
        element.text for element in soup.find_all(TEXT_TAGS) if not element_is_dispensable(element)
    ])
