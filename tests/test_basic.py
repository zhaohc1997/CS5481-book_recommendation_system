import unittest
import sys
import os

sys.path.append('src')

from data_processing.data_loader import DataLoader
from data_processing.data_cleaner import DataCleaner


class TestBookRecommendation(unittest.TestCase):

    def test_data_loading(self):
        """测试数据加载"""
        loader = DataLoader('../data/raw/')
        books = loader.load_books_data()
        self.assertIsNotNone(books)
        self.assertTrue(len(books) > 0)

    def test_data_cleaning(self):
        """测试数据清洗"""
        loader = DataLoader('../data/raw/')
        cleaner = DataCleaner()

        books = loader.load_books_data()
        cleaned_books = cleaner.clean_books_data(books)

        self.assertTrue(len(cleaned_books) <= len(books))


if __name__ == '__main__':
    unittest.main()