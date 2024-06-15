import {
  to_words_le,
  from_words_le,
  compute_misc_params,
  bigints_to_u8_for_gpu,
} from "../../implementation/cuzk/utils";
import * as bigintCryptoUtils from "bigint-crypto-utils";
import { bigints_to_16_bit_words_for_gpu } from "../utils";

describe("utils", () => {
  const p = BigInt(
    "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
  );
  const word_size = 13;
  const num_words = 20;

  const testData: [bigint, Uint16Array, number][] = [
    [
      p,
      new Uint16Array([
        1, 0, 0, 768, 4257, 0, 0, 8154, 2678, 2765, 3072, 6255, 4581, 6694,
        6530, 5290, 6700, 2804, 2777, 37,
      ]),
      word_size,
    ],
  ];

  describe("bigint to limbs and vice versa", () => {
    it.each(testData)(
      "to_words_le",
      (val: bigint, expected: Uint16Array, word_size: number) => {
        const words = to_words_le(val, num_words, word_size);
        expect(words).toEqual(expected);
      },
    );

    it.each(testData)(
      "from_words_le",
      (expected: bigint, words: Uint16Array, word_size: number) => {
        const val = from_words_le(words, num_words, word_size);
        expect(val).toEqual(expected);
      },
    );

    /*
     * @param input A big-endian buffer of bytes
     * @param word_idx The index of the word_size word you want to
     *                 extract. This index is of a little-endian array.
     * @param word_size The bitlength of each word to extract. It should be greater than 8.
     */
    const extract_word_from_bytes_le = (
      input: Buffer,
      word_idx: number,
      word_size: number,
    ): number => {
      /*
       * 3333 3333 2222 2222 1111 1111 0000 0000
       * .... ..-- ---- ---- ---. .... .... ....
       */
      const start_byte_idx =
        input.length - 1 - Math.floor((word_idx * word_size + word_size) / 8);
      const end_byte_idx =
        input.length - 1 - Math.floor((word_idx * word_size) / 8);

      const start_byte_offset = (word_idx * word_size + word_size) % 8;
      const end_byte_offset = (word_idx * word_size) % 8;

      let sum = 0;
      for (let i = start_byte_idx; i <= end_byte_idx; i++) {
        if (i === start_byte_idx) {
          const mask = 2 ** start_byte_offset - 1;
          sum += input[i] & mask;
        } else if (i === end_byte_idx) {
          sum = sum << (8 - end_byte_offset);
          sum += input[i] >> end_byte_offset;
        } else {
          sum = sum << 8;
          sum += input[i];
        }
      }

      return sum;
    };

    const to_words_le_modified = (
      val: bigint,
      num_words: number,
      word_size: number,
    ): Uint16Array => {
      let hex_str = val.toString(16);
      while (hex_str.length < Math.ceil((num_words * word_size) / 8) * 2) {
        hex_str = "0" + hex_str;
      }
      const bytes = Buffer.from(hex_str, "hex");

      const words = Array(num_words).fill(0);
      for (let i = 0; i < num_words; i++) {
        const w = extract_word_from_bytes_le(bytes, i, word_size);
        words[i] = w;
      }
      return new Uint16Array(words);
    };

    it("bytes to words", () => {
      const test_cases = [
        BigInt(0),
        BigInt(1),
        BigInt("0x10"),
        BigInt("0xabcde"),
        BigInt("0xffffffffffffffff"),
        p,
      ];
      for (const test_case of test_cases) {
        const words = to_words_le_modified(test_case, num_words, word_size);
        expect(words.toString()).toEqual(
          to_words_le(test_case, num_words, word_size).toString(),
        );
      }
    });

    it("bigints to 16-bit words", () => {
      const test_cases = [
        [p],
        [p, p],
        [p, p, p],
        [p, p, p, p],
        [BigInt(0)],
        [BigInt(0), BigInt(0)],
        [BigInt(1)],
        [BigInt(1), BigInt(1)],
        [BigInt(255)],
        [BigInt(255), BigInt(255)],
        [BigInt(1), BigInt(255)],
        [BigInt(0), BigInt(255)],
        [BigInt(255), BigInt(0)],
        [p, p / BigInt(2), p / BigInt(3), p],
      ];
      for (const vals of test_cases) {
        const b = bigints_to_16_bit_words_for_gpu(vals);
        const c = bigints_to_u8_for_gpu(vals);
        expect(b.toString()).toEqual(c.toString());
      }
    });
  });

  describe("misc functions", () => {
    it("compute_misc_params", () => {
      // Check that there are enough limbs to fit the modulus
      expect(2 ** (num_words * word_size)).toBeGreaterThanOrEqual(p);

      // The Montgomery radix
      const r = BigInt(2) ** BigInt(num_words * word_size);

      // Returns triple (g, rinv, pprime)
      const egcdResult = bigintCryptoUtils.eGcd(r, p);

      // Sanity-check rinv and pprime
      if (egcdResult.x < BigInt(0)) {
        expect(((r * egcdResult.x - p * egcdResult.y) % p) + BigInt(p)).toEqual(
          BigInt(1),
        );
        expect((r * egcdResult.x) % p == BigInt(1));
        expect((p * egcdResult.y) % r == BigInt(1));
      } else {
        expect((r * egcdResult.x - p * egcdResult.y) % p).toEqual(BigInt(1));
        expect((r * egcdResult.x) % p == BigInt(1));
        expect((p * egcdResult.y) % r == BigInt(1));
      }

      const misc = compute_misc_params(p, word_size);
      const expected = {
        num_words,
        max_terms: 40,
        k: 65,
        nsafe: 32,
        n0: BigInt(8191),
        r: r % p,
        rinv: misc.rinv,
      };
      for (const k of Object.keys(expected)) {
        expect((misc as any)[k]).toEqual((expected as any)[k]);
      }
    });
  });
});
