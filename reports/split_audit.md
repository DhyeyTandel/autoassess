# Train/test near-duplicate audit

## cardd

**Method:** phash (64-bit DCT), nearest train image per test image, flagged when Hamming distance <= 6.

- Train images hashed: 2816
- Test images hashed: 374
- Unreadable files excluded: 0
- Test images with a train neighbour at distance <= 6: 0 (0.0%)
- Flagged test image list: [cardd_flagged_test.txt](split_audit/cardd_flagged_test.txt) (usable as `autoassess-eval --exclude-list`)

Nearest-neighbour distance histogram (test images per bin):

| Distance | Test images |
|---|---|
| 0 | 0 |
| 1-3 | 0 |
| 4-6 | 0 |
| 7-10 | 1 |
| 11-20 | 373 |
| >20 | 0 |

Top 30 closest pairs:

| Rank | Test | Train | Distance | Figure |
|---|---|---|---|---|
| 1 | 002810.jpg | 002864.jpg | 10 | [pair_01_d10.jpg](figures/split_audit/cardd/pair_01_d10.jpg) |
| 2 | 001168.jpg | 003077.jpg | 12 | [pair_02_d12.jpg](figures/split_audit/cardd/pair_02_d12.jpg) |
| 3 | 001558.jpg | 003674.jpg | 12 | [pair_03_d12.jpg](figures/split_audit/cardd/pair_03_d12.jpg) |
| 4 | 001720.jpg | 003538.jpg | 12 | [pair_04_d12.jpg](figures/split_audit/cardd/pair_04_d12.jpg) |
| 5 | 001843.jpg | 000457.jpg | 12 | [pair_05_d12.jpg](figures/split_audit/cardd/pair_05_d12.jpg) |
| 6 | 002045.jpg | 001443.jpg | 12 | [pair_06_d12.jpg](figures/split_audit/cardd/pair_06_d12.jpg) |
| 7 | 002483.jpg | 001854.jpg | 12 | [pair_07_d12.jpg](figures/split_audit/cardd/pair_07_d12.jpg) |
| 8 | 002814.jpg | 002217.jpg | 12 | [pair_08_d12.jpg](figures/split_audit/cardd/pair_08_d12.jpg) |
| 9 | 002879.jpg | 002417.jpg | 12 | [pair_09_d12.jpg](figures/split_audit/cardd/pair_09_d12.jpg) |
| 10 | 003831.jpg | 000133.jpg | 12 | [pair_10_d12.jpg](figures/split_audit/cardd/pair_10_d12.jpg) |
| 11 | 000015.jpg | 001227.jpg | 14 | [pair_11_d14.jpg](figures/split_audit/cardd/pair_11_d14.jpg) |
| 12 | 000023.jpg | 003936.jpg | 14 | [pair_12_d14.jpg](figures/split_audit/cardd/pair_12_d14.jpg) |
| 13 | 000320.jpg | 002658.jpg | 14 | [pair_13_d14.jpg](figures/split_audit/cardd/pair_13_d14.jpg) |
| 14 | 000424.jpg | 003401.jpg | 14 | [pair_14_d14.jpg](figures/split_audit/cardd/pair_14_d14.jpg) |
| 15 | 000466.jpg | 000712.jpg | 14 | [pair_15_d14.jpg](figures/split_audit/cardd/pair_15_d14.jpg) |
| 16 | 000468.jpg | 003056.jpg | 14 | [pair_16_d14.jpg](figures/split_audit/cardd/pair_16_d14.jpg) |
| 17 | 000603.jpg | 001048.jpg | 14 | [pair_17_d14.jpg](figures/split_audit/cardd/pair_17_d14.jpg) |
| 18 | 000625.jpg | 002472.jpg | 14 | [pair_18_d14.jpg](figures/split_audit/cardd/pair_18_d14.jpg) |
| 19 | 000687.jpg | 003633.jpg | 14 | [pair_19_d14.jpg](figures/split_audit/cardd/pair_19_d14.jpg) |
| 20 | 000914.jpg | 001858.jpg | 14 | [pair_20_d14.jpg](figures/split_audit/cardd/pair_20_d14.jpg) |
| 21 | 001024.jpg | 000382.jpg | 14 | [pair_21_d14.jpg](figures/split_audit/cardd/pair_21_d14.jpg) |
| 22 | 001098.jpg | 000743.jpg | 14 | [pair_22_d14.jpg](figures/split_audit/cardd/pair_22_d14.jpg) |
| 23 | 001268.jpg | 000806.jpg | 14 | [pair_23_d14.jpg](figures/split_audit/cardd/pair_23_d14.jpg) |
| 24 | 001282.jpg | 003148.jpg | 14 | [pair_24_d14.jpg](figures/split_audit/cardd/pair_24_d14.jpg) |
| 25 | 001409.jpg | 001019.jpg | 14 | [pair_25_d14.jpg](figures/split_audit/cardd/pair_25_d14.jpg) |
| 26 | 001488.jpg | 002099.jpg | 14 | [pair_26_d14.jpg](figures/split_audit/cardd/pair_26_d14.jpg) |
| 27 | 001524.jpg | 003135.jpg | 14 | [pair_27_d14.jpg](figures/split_audit/cardd/pair_27_d14.jpg) |
| 28 | 001746.jpg | 002300.jpg | 14 | [pair_28_d14.jpg](figures/split_audit/cardd/pair_28_d14.jpg) |
| 29 | 001820.jpg | 002387.jpg | 14 | [pair_29_d14.jpg](figures/split_audit/cardd/pair_29_d14.jpg) |
| 30 | 001874.jpg | 002165.jpg | 14 | [pair_30_d14.jpg](figures/split_audit/cardd/pair_30_d14.jpg) |

**How to read this.** phash flags visually similar photos, not proven leakage: open the figures and review the closest pairs by eye to decide whether they are the same photo or vehicle. Same-vehicle shots taken from a different angle or distance will NOT be caught, so a low flagged count is not proof of a clean split.

## carparts

**Method:** phash (64-bit DCT), nearest train image per test image, flagged when Hamming distance <= 6.

- Train images hashed: 3156
- Test images hashed: 276
- Unreadable files excluded: 0
- Test images with a train neighbour at distance <= 6: 12 (4.3%)
- Flagged test image list: [carparts_flagged_test.txt](split_audit/carparts_flagged_test.txt) (usable as `autoassess-eval --exclude-list`)

Nearest-neighbour distance histogram (test images per bin):

| Distance | Test images |
|---|---|
| 0 | 1 |
| 1-3 | 2 |
| 4-6 | 9 |
| 7-10 | 20 |
| 11-20 | 244 |
| >20 | 0 |

Top 30 closest pairs:

| Rank | Test | Train | Distance | Figure |
|---|---|---|---|---|
| 1 | train33_jpg.rf.443a660fd96c1a39bd67dc0d95abc529.jpg | train67_jpg.rf.58f658e5abda85ac6b7de40c8ade595b.jpg | 0 | [pair_01_d0.jpg](figures/split_audit/carparts/pair_01_d0.jpg) |
| 2 | train109_jpg.rf.08dba575830134e01840fcf86651a0f1.jpg | train11_jpg.rf.d2e91dfc0d9526e9b38a5684f4905b39.jpg | 2 | [pair_02_d2.jpg](figures/split_audit/carparts/pair_02_d2.jpg) |
| 3 | train385_jpg.rf.829acf83603d294e6ab4d16d7d58ef46.jpg | train385_jpg.rf.1f26ca0a1f6648158d674812c505ffd5.jpg | 2 | [pair_03_d2.jpg](figures/split_audit/carparts/pair_03_d2.jpg) |
| 4 | car4_jpg.rf.8978131a7b03be689c244641e42e1307.jpg | car4_jpg.rf.fb888382a8d302f407ed8d163c647730.jpg | 4 | [pair_04_d4.jpg](figures/split_audit/carparts/pair_04_d4.jpg) |
| 5 | train102_jpg.rf.35a1739cad86d20f2c125194253f60bd.jpg | train102_jpg.rf.f1cec9feb442de92d7b3e9a12adeabb7.jpg | 4 | [pair_05_d4.jpg](figures/split_audit/carparts/pair_05_d4.jpg) |
| 6 | train188_jpg.rf.649354ed3bc34d0deda30305fe95b153.jpg | train188_jpg.rf.5aab4183fe443d40fb042268d62c7ea3.jpg | 4 | [pair_06_d4.jpg](figures/split_audit/carparts/pair_06_d4.jpg) |
| 7 | train352_jpg.rf.d427037b79fa65d940f6a659674b702c.jpg | train352_jpg.rf.24ea8dd17b5f196711e5cd0f726ba9fa.jpg | 4 | [pair_07_d4.jpg](figures/split_audit/carparts/pair_07_d4.jpg) |
| 8 | te95_jpg.rf.90e0581218f8ab3c48a779dd976bdd62.jpg | te95_jpg.rf.76ac7db9f115ba03b55513d23f82127f.jpg | 6 | [pair_08_d6.jpg](figures/split_audit/carparts/pair_08_d6.jpg) |
| 9 | train266_jpg.rf.075be2ee75dd2a38767eb96d12b2dc3d.jpg | train266_jpg.rf.5a070442ec80e9dfb963bd8d1342bbae.jpg | 6 | [pair_09_d6.jpg](figures/split_audit/carparts/pair_09_d6.jpg) |
| 10 | train326_jpg.rf.e0b1e35f9ca4c2f27b1824cca83c540b.jpg | train326_jpg.rf.022ebd4898acd66531e5d89684cba656.jpg | 6 | [pair_10_d6.jpg](figures/split_audit/carparts/pair_10_d6.jpg) |
| 11 | train35_jpg.rf.5dcc32ab156353262ee723907e74f6b3.jpg | train35_jpg.rf.0a116a9f0d4f350d250bd41364c4418e.jpg | 6 | [pair_11_d6.jpg](figures/split_audit/carparts/pair_11_d6.jpg) |
| 12 | train56_jpg.rf.ce7baacc457a3783f8e12885d783e654.jpg | train56_jpg.rf.a48b0095637459949db740a0eda0b69a.jpg | 6 | [pair_12_d6.jpg](figures/split_audit/carparts/pair_12_d6.jpg) |
| 13 | te29_jpg.rf.6a350f4c5008d8fde57b1663b90e3ee6.jpg | te29_jpg.rf.132ab0e9ac9c35a0be57506fc44a750e.jpg | 8 | [pair_13_d8.jpg](figures/split_audit/carparts/pair_13_d8.jpg) |
| 14 | te99_jpg.rf.8de3a61388e0ad908f547232f432d43a.jpg | te99_jpg.rf.a4ce6ee977b55bf2b0406c38d182148a.jpg | 8 | [pair_14_d8.jpg](figures/split_audit/carparts/pair_14_d8.jpg) |
| 15 | train119_jpg.rf.5361d61671a47d78760a2dbba1b69019.jpg | train117_jpg.rf.356c57185d065c0f246a1bff17dd64d8.jpg | 8 | [pair_15_d8.jpg](figures/split_audit/carparts/pair_15_d8.jpg) |
| 16 | train162_jpg.rf.32b24e39e9ba058cc8fcdc4b7a4e1151.jpg | train201_jpg.rf.0403c8bf16417588c77037dbfcc2fa43.jpg | 8 | [pair_16_d8.jpg](figures/split_audit/carparts/pair_16_d8.jpg) |
| 17 | train294_jpg.rf.4f0f2e391c321b991a15a2decfd996b5.jpg | train294_jpg.rf.04091115d72cff4ab42924ad50f04a90.jpg | 8 | [pair_17_d8.jpg](figures/split_audit/carparts/pair_17_d8.jpg) |
| 18 | train39_jpg.rf.7cba5ecc4877bb9dd754ce853d47068f.jpg | train39_jpg.rf.e5ae465840d8431e3b47e0acccf7c601.jpg | 8 | [pair_18_d8.jpg](figures/split_audit/carparts/pair_18_d8.jpg) |
| 19 | new_33_png_jpg.rf.ae8a3978a575bcae704633a0f075a6e1.jpg | new_33_png_jpg.rf.6a98833eac45f31cdaabf05833e3e777.jpg | 10 | [pair_19_d10.jpg](figures/split_audit/carparts/pair_19_d10.jpg) |
| 20 | train129_jpg.rf.e1a5999b3630cbbcd9fff3555b832eb0.jpg | train377_jpg.rf.cde251695f846037e9b94eea97e59047.jpg | 10 | [pair_20_d10.jpg](figures/split_audit/carparts/pair_20_d10.jpg) |
| 21 | train140_jpg.rf.4f4623cdce2d20e7443013d6620206cd.jpg | train140_jpg.rf.897d78bad75d5fb1ee1441ebed3a8629.jpg | 10 | [pair_21_d10.jpg](figures/split_audit/carparts/pair_21_d10.jpg) |
| 22 | train210_jpg.rf.c0cf0ad9662d1962dd9e23b6f33560a3.jpg | train210_jpg.rf.d465e6b9b60a71bfabd40a6dc7519440.jpg | 10 | [pair_22_d10.jpg](figures/split_audit/carparts/pair_22_d10.jpg) |
| 23 | train248_jpg.rf.d5ca84d72be0b763aa01fbdcad4d4af4.jpg | train248_jpg.rf.3bdc358861e07c76eeb948e5104d4f60.jpg | 10 | [pair_23_d10.jpg](figures/split_audit/carparts/pair_23_d10.jpg) |
| 24 | train298_jpg.rf.a47f2ec270a859617283eb6b3762aedc.jpg | train298_jpg.rf.14fee71fbd39c4aea2346a240901d250.jpg | 10 | [pair_24_d10.jpg](figures/split_audit/carparts/pair_24_d10.jpg) |
| 25 | train331_jpg.rf.7a52c16d1bfd9d32eef966b2a0699c82.jpg | train331_jpg.rf.bb9fa60d7ee60b83c87f3fa16da3bdab.jpg | 10 | [pair_25_d10.jpg](figures/split_audit/carparts/pair_25_d10.jpg) |
| 26 | train350_jpg.rf.0609e8980418e890f807920cfdf855b2.jpg | te81_jpg.rf.5c02d1720144f0359062fd7a5db969d5.jpg | 10 | [pair_26_d10.jpg](figures/split_audit/carparts/pair_26_d10.jpg) |
| 27 | train372_jpg.rf.583d5372a90abeb5c30814fa234a1fee.jpg | train372_jpg.rf.37cd3efb586ef442fd51760ac77e7446.jpg | 10 | [pair_27_d10.jpg](figures/split_audit/carparts/pair_27_d10.jpg) |
| 28 | train37_jpg.rf.38292194304ff944402b5e9f2f91e3a3.jpg | train37_jpg.rf.2c44e9933ea71306780e222e69e8d742.jpg | 10 | [pair_28_d10.jpg](figures/split_audit/carparts/pair_28_d10.jpg) |
| 29 | train384_jpg.rf.6558520cf31e6a07323e8213a6748b4b.jpg | train384_jpg.rf.ed466e1d640929636dbf5f69180c6055.jpg | 10 | [pair_29_d10.jpg](figures/split_audit/carparts/pair_29_d10.jpg) |
| 30 | train42_jpg.rf.642a2e31c3933551646e392becdf7918.jpg | train37_jpg.rf.551e27a7dbf2a8e602b802bf7f6a45d3.jpg | 10 | [pair_30_d10.jpg](figures/split_audit/carparts/pair_30_d10.jpg) |

**How to read this.** phash flags visually similar photos, not proven leakage: open the figures and review the closest pairs by eye to decide whether they are the same photo or vehicle. Same-vehicle shots taken from a different angle or distance will NOT be caught, so a low flagged count is not proof of a clean split.

## vehide

**Method:** phash (64-bit DCT), nearest train image per test image, flagged when Hamming distance <= 6.

- Train images hashed: 9866
- Test images hashed: 1741
- Unreadable files excluded: 14
- Test images with a train neighbour at distance <= 6: 45 (2.6%)
- Flagged test image list: [vehide_flagged_test.txt](split_audit/vehide_flagged_test.txt) (usable as `autoassess-eval --exclude-list`)

Nearest-neighbour distance histogram (test images per bin):

| Distance | Test images |
|---|---|
| 0 | 43 |
| 1-3 | 0 |
| 4-6 | 2 |
| 7-10 | 23 |
| 11-20 | 1673 |
| >20 | 0 |

Unreadable files (excluded from every count above):

- `images/test/25032020_091214image992948.jpg`
- `images/test/25032020_091232image529852.jpg`
- `images/train/02012020_082351image833616.jpg`
- `images/train/04052020_152057image59498.jpg`
- `images/train/04052020_152101image628633.jpg`
- `images/train/04052020_152107image748519.jpg`
- `images/train/13032020_144737image20659.jpg`
- `images/train/13032020_144742image419520.jpg`
- `images/train/13032020_144753image617184.jpg`
- `images/train/13032020_144756image760735.jpg`
- `images/train/13032020_144759image376956.jpg`
- `images/train/24032020_091653image573732.jpg`
- `images/train/24032020_140444image375775.jpg`
- `images/train/28042020_081842image605326.jpg`

Top 30 closest pairs:

| Rank | Test | Train | Distance | Figure |
|---|---|---|---|---|
| 1 | 02012020_095025image124982.jpg | Thumbnail02012020_095025image124982.jpg | 0 | [pair_01_d0.jpg](figures/split_audit/vehide/pair_01_d0.jpg) |
| 2 | 03012020_092135image313562.jpg | 4ef31503012020_092135image313562.jpg | 0 | [pair_02_d0.jpg](figures/split_audit/vehide/pair_02_d0.jpg) |
| 3 | 09012020_163938image571442.jpg | 37669409012020_163938image571442.jpg | 0 | [pair_03_d0.jpg](figures/split_audit/vehide/pair_03_d0.jpg) |
| 4 | 11bb5372305ac804914b.jpg | Thumbnail11bb5372305ac804914b.jpg | 0 | [pair_04_d0.jpg](figures/split_audit/vehide/pair_04_d0.jpg) |
| 5 | 3c139c978ec876962fd9.jpg | Thumbnail3c139c978ec876962fd9.jpg | 0 | [pair_05_d0.jpg](figures/split_audit/vehide/pair_05_d0.jpg) |
| 6 | 8d8b943ecc5e37006e4f.jpg | Thumbnail8d8b943ecc5e37006e4f.jpg | 0 | [pair_06_d0.jpg](figures/split_audit/vehide/pair_06_d0.jpg) |
| 7 | Thumbnail02012020_095009image943782.jpg | 02012020_095009image943782.jpg | 0 | [pair_07_d0.jpg](figures/split_audit/vehide/pair_07_d0.jpg) |
| 8 | Thumbnail17.jpg | 17.jpg | 0 | [pair_08_d0.jpg](figures/split_audit/vehide/pair_08_d0.jpg) |
| 9 | Thumbnail18.jpg | 18.jpg | 0 | [pair_09_d0.jpg](figures/split_audit/vehide/pair_09_d0.jpg) |
| 10 | Thumbnail45a0eab401c7f999a0d6.jpg | 45a0eab401c7f999a0d6.jpg | 0 | [pair_10_d0.jpg](figures/split_audit/vehide/pair_10_d0.jpg) |
| 11 | ThumbnailH_(12).jpg | H_(12).jpg | 0 | [pair_11_d0.jpg](figures/split_audit/vehide/pair_11_d0.jpg) |
| 12 | Thumbnailb2bcc206acd2548c0dc3.jpg | b2bcc206acd2548c0dc3.jpg | 0 | [pair_12_d0.jpg](figures/split_audit/vehide/pair_12_d0.jpg) |
| 13 | Thumbnailc2243141cd72352c6c63.jpg | c2243141cd72352c6c63.jpg | 0 | [pair_13_d0.jpg](figures/split_audit/vehide/pair_13_d0.jpg) |
| 14 | Thumbnaildac7c95c9f1b67453e0a.jpg | dac7c95c9f1b67453e0a.jpg | 0 | [pair_14_d0.jpg](figures/split_audit/vehide/pair_14_d0.jpg) |
| 15 | Thumbnailfc446c2341fdb9a3e0ec.jpg | fc446c2341fdb9a3e0ec.jpg | 0 | [pair_15_d0.jpg](figures/split_audit/vehide/pair_15_d0.jpg) |
| 16 | Thumbnailz1635882241294_8f6901715bbc0ec50a3aacf52c10429f.jpg | z1635882241294_8f6901715bbc0ec50a3aacf52c10429f.jpg | 0 | [pair_16_d0.jpg](figures/split_audit/vehide/pair_16_d0.jpg) |
| 17 | Thumbnailz1689258876799_b8c76aebd867f17fab7fc0535e2ad770.jpg | z1689258876799_b8c76aebd867f17fab7fc0535e2ad770.jpg | 0 | [pair_17_d0.jpg](figures/split_audit/vehide/pair_17_d0.jpg) |
| 18 | Thumbnailz1691631592173_cf6e832bbd319a99d7cb520d87340a3d.jpg | z1691631592173_cf6e832bbd319a99d7cb520d87340a3d.jpg | 0 | [pair_18_d0.jpg](figures/split_audit/vehide/pair_18_d0.jpg) |
| 19 | Thumbnailz1693968528157_0180b43efe2ae930605f4f2024c9261c.jpg | z1693968528157_0180b43efe2ae930605f4f2024c9261c.jpg | 0 | [pair_19_d0.jpg](figures/split_audit/vehide/pair_19_d0.jpg) |
| 20 | Thumbnailz1701323111724_374147332e32f5e4e8b0d64d18ed06ce.jpg | z1701323111724_374147332e32f5e4e8b0d64d18ed06ce.jpg | 0 | [pair_20_d0.jpg](figures/split_audit/vehide/pair_20_d0.jpg) |
| 21 | Thumbnailz1707671103073_48c4d4d414c4ea3eddc2d1675a134ebd.jpg | z1707671103073_48c4d4d414c4ea3eddc2d1675a134ebd.jpg | 0 | [pair_21_d0.jpg](figures/split_audit/vehide/pair_21_d0.jpg) |
| 22 | Thumbnailz1707767302385_dd509984de96ab5afaac01243288b2b1.jpg | z1707767302385_dd509984de96ab5afaac01243288b2b1.jpg | 0 | [pair_22_d0.jpg](figures/split_audit/vehide/pair_22_d0.jpg) |
| 23 | Thumbnailz2116887754580_ff0edc2ef3d469ad758a6e9a649248ac.jpg | z2116887754580_ff0edc2ef3d469ad758a6e9a649248ac.jpg | 0 | [pair_23_d0.jpg](figures/split_audit/vehide/pair_23_d0.jpg) |
| 24 | Thumbnailz2126774386416_0512cf6a297d14e1f42f9b43e56dff11.jpg | z2126774386416_0512cf6a297d14e1f42f9b43e56dff11.jpg | 0 | [pair_24_d0.jpg](figures/split_audit/vehide/pair_24_d0.jpg) |
| 25 | Thumbnailz2129091240897_dc91869231a4aa4a215450327cf3f0ce.jpg | z2129091240897_dc91869231a4aa4a215450327cf3f0ce.jpg | 0 | [pair_25_d0.jpg](figures/split_audit/vehide/pair_25_d0.jpg) |
| 26 | Thumbnailz2148990058334_ed74cd6597997e060ea932c2d67a2af6.jpg | z2148990058334_ed74cd6597997e060ea932c2d67a2af6.jpg | 0 | [pair_26_d0.jpg](figures/split_audit/vehide/pair_26_d0.jpg) |
| 27 | b3c318965ee0a6befff1.jpg | Thumbnailb3c318965ee0a6befff1.jpg | 0 | [pair_27_d0.jpg](figures/split_audit/vehide/pair_27_d0.jpg) |
| 28 | bd7f08575afea2a0fbef.jpg | Thumbnailbd7f08575afea2a0fbef.jpg | 0 | [pair_28_d0.jpg](figures/split_audit/vehide/pair_28_d0.jpg) |
| 29 | d5f85a61f0a20bfc52b3.jpg | Thumbnaild5f85a61f0a20bfc52b3.jpg | 0 | [pair_29_d0.jpg](figures/split_audit/vehide/pair_29_d0.jpg) |
| 30 | e3917d26102020_112015image817530.jpg | 26102020_112015image817530.jpg | 0 | [pair_30_d0.jpg](figures/split_audit/vehide/pair_30_d0.jpg) |

**How to read this.** phash flags visually similar photos, not proven leakage: open the figures and review the closest pairs by eye to decide whether they are the same photo or vehicle. Same-vehicle shots taken from a different angle or distance will NOT be caught, so a low flagged count is not proof of a clean split.
