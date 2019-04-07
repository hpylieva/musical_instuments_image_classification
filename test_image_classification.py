import unittest

from classify_images import *


class TestDataLoader(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        df = pd.read_excel('products.xlsx').dropna(subset=['category', 'condition'])
        y = pd.get_dummies(df['category']).values
        ids = df['id'].values
        self.dataset = ImageData('.', 'img_n', ids, y)
        self.img, self.label = self.dataset[0]

    def test_im_loaded(self):
        self.assertIsNotNone(self.img)

    def test_label_loaded(self):
        self.assertIsNotNone(self.label)


if __name__ == '__main__':
    unittest.main()
