# Third-Party Notices

BrickAgain depends on external datasets, model artifacts, formats, and
reference code. The BrickAgain MIT License applies only to original
BrickAgain source code; it cannot replace any third-party terms below.

## BrickGPT

- Project: <https://github.com/AvaLovelace1/BrickGPT>
- Authors: Ava Pun, Kangle Deng, Ruixuan Liu, Deva Ramanan, Changliu Liu, and
  Jun-Yan Zhu
- Paper: *Generating Physically Stable and Buildable Brick Structures from
  Text*, ICCV 2025, <https://arxiv.org/abs/2505.05469>
- Upstream license: MIT

BrickAgain reuses or adapts BrickGPT's instruction wording, LDraw coordinate
conversion, and basic-part identifier mapping. Those portions remain subject
to the upstream MIT license and notice:

```text
MIT License

Copyright (c) 2025 Ava Pun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## StableText2Brick

- Dataset: <https://huggingface.co/datasets/AvaLovelace/StableText2Brick>
- Associated BrickGPT paper: <https://arxiv.org/abs/2505.05469>
- Dataset card license: MIT

The upstream dataset is downloaded separately and is not committed to this
repository. The frozen split manifest and aggregate reports in this repository
were derived from it. Users remain responsible for reviewing the current
dataset card and license before downloading or redistributing the data.

## Llama 3.2

- Base model: `meta-llama/Llama-3.2-1B-Instruct`
- License: <https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE>

Llama 3.2 is subject to Meta's Llama 3.2 Community License rather than the MIT
License. BrickAgain references the model identifier but does not include or
redistribute Llama weights, BrickGPT adapter weights, or locally trained
adapter weights. Users must obtain their own access and comply with the
applicable license. Publishing a trained adapter or other model derivative
requires a separate license review and is outside the scope of this source-code
release preparation.

## LDraw

- Project and format: <https://www.ldraw.org/>
- File-format specification: <https://www.ldraw.org/article/218.html>
- Parts library: <https://library.ldraw.org/>

BrickAgain writes LDraw-compatible text files. The LDraw parts library is not
included. Users who install or redistribute that library must follow its own
license and attribution terms.
