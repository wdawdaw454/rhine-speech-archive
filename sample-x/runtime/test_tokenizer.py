"""Regression coverage for literal added tokens mixed with byte-level BPE."""
from pathlib import Path
import unittest
from native_runtime import Tokenizer


TOKENIZER_ASSET = Path(__file__).resolve().parent / 'decoded/asr/token/vocab.txt'


@unittest.skipUnless(TOKENIZER_ASSET.is_file(), 'Sample-X tokenizer assets are not redistributed')
class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=Tokenizer()

    def test_chinese_numerals(self):
        self.assertEqual(self.tokenizer.decode(range(17,30)), '一二三四五六七八九十百千万')

    def test_mixed_added_and_byte_tokens(self):
        t=self.tokenizer
        ids=t.encode('明确写有')+[20]+t.encode('种语言，')+[8,7]+t.encode('秒。')
        self.assertEqual(t.decode(ids), '明确写有四种语言，10秒。')

    def test_utf8_bytes_can_span_tokens(self):
        t=self.tokenizer
        ids=[t.vocab[t.enc[b]] for b in '中文🙂'.encode('utf8')]
        self.assertEqual(t.decode(ids), '中文🙂')

    def test_every_vocab_token_decodes(self):
        for token in self.tokenizer.inverse:
            self.tokenizer.decode([token])


if __name__=='__main__':unittest.main(verbosity=2)
