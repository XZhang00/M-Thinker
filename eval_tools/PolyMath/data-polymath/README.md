---
configs:
- config_name: ar
  data_files:
    - split: top
      path: ar/top.parquet
    - split: high
      path: ar/high.parquet
    - split: medium
      path: ar/medium.parquet
    - split: low
      path: ar/low.parquet
- config_name: bn
  data_files:
    - split: top
      path: bn/top.parquet
    - split: high
      path: bn/high.parquet
    - split: medium
      path: bn/medium.parquet
    - split: low
      path: bn/low.parquet
- config_name: de
  data_files:
    - split: top
      path: de/top.parquet
    - split: high
      path: de/high.parquet
    - split: medium
      path: de/medium.parquet
    - split: low
      path: de/low.parquet
- config_name: en
  data_files:
    - split: top
      path: en/top.parquet
    - split: high
      path: en/high.parquet
    - split: medium
      path: en/medium.parquet
    - split: low
      path: en/low.parquet
- config_name: es
  data_files:
    - split: top
      path: es/top.parquet
    - split: high
      path: es/high.parquet
    - split: medium
      path: es/medium.parquet
    - split: low
      path: es/low.parquet
- config_name: fr
  data_files:
    - split: top
      path: fr/top.parquet
    - split: high
      path: fr/high.parquet
    - split: medium
      path: fr/medium.parquet
    - split: low
      path: fr/low.parquet
- config_name: id
  data_files:
    - split: top
      path: id/top.parquet
    - split: high
      path: id/high.parquet
    - split: medium
      path: id/medium.parquet
    - split: low
      path: id/low.parquet
- config_name: it
  data_files:
    - split: top
      path: it/top.parquet
    - split: high
      path: it/high.parquet
    - split: medium
      path: it/medium.parquet
    - split: low
      path: it/low.parquet
- config_name: ja
  data_files:
    - split: top
      path: ja/top.parquet
    - split: high
      path: ja/high.parquet
    - split: medium
      path: ja/medium.parquet
    - split: low
      path: ja/low.parquet
- config_name: ko
  data_files:
    - split: top
      path: ko/top.parquet
    - split: high
      path: ko/high.parquet
    - split: medium
      path: ko/medium.parquet
    - split: low
      path: ko/low.parquet
- config_name: ms
  data_files:
    - split: top
      path: ms/top.parquet
    - split: high
      path: ms/high.parquet
    - split: medium
      path: ms/medium.parquet
    - split: low
      path: ms/low.parquet
- config_name: pt
  data_files:
    - split: top
      path: pt/top.parquet
    - split: high
      path: pt/high.parquet
    - split: medium
      path: pt/medium.parquet
    - split: low
      path: pt/low.parquet
- config_name: ru
  data_files:
    - split: top
      path: ru/top.parquet
    - split: high
      path: ru/high.parquet
    - split: medium
      path: ru/medium.parquet
    - split: low
      path: ru/low.parquet
- config_name: sw
  data_files:
    - split: top
      path: sw/top.parquet
    - split: high
      path: sw/high.parquet
    - split: medium
      path: sw/medium.parquet
    - split: low
      path: sw/low.parquet
- config_name: te
  data_files:
    - split: top
      path: te/top.parquet
    - split: high
      path: te/high.parquet
    - split: medium
      path: te/medium.parquet
    - split: low
      path: te/low.parquet
- config_name: th
  data_files:
    - split: top
      path: th/top.parquet
    - split: high
      path: th/high.parquet
    - split: medium
      path: th/medium.parquet
    - split: low
      path: th/low.parquet
- config_name: vi
  data_files:
    - split: top
      path: vi/top.parquet
    - split: high
      path: vi/high.parquet
    - split: medium
      path: vi/medium.parquet
    - split: low
      path: vi/low.parquet
- config_name: zh
  data_files:
    - split: top
      path: zh/top.parquet
    - split: high
      path: zh/high.parquet
    - split: medium
      path: zh/medium.parquet
    - split: low
      path: zh/low.parquet
---




<div align="center">

  <h2>
    PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts
  </h2>

</div>



<div align="center">
  <a href="https://arxiv.org/abs/2504.18428">
    <img src="https://img.shields.io/badge/arXiv-2504.18428-b31b1b.svg?logo=arxiv" alt="arXiv Badge"/>
    </a>
    <a href="https://github.com/QwenLM/PolyMath">
    <img src="https://img.shields.io/badge/GitHub-Code-black?logo=github" alt="GitHub Badge"/>
</a>
</div>



**PolyMath** is a multilingual mathematical reasoning benchmark **covering 18 languages** and **4 easy-to-hard difficulty levels**. Our benchmark ensures *difficulty comprehensiveness*, *language diversity*, and *high-quality translation*, making it a highly discriminative multilingual mathematical benchmark in the era of reasoning LLMs.


- 📈 **Broad Difficulty Range:** PolyMath defines and partitions mathematical difficulty across four levels using two core dimensions: **Thought Depth** and **Knowledge Breadth**, ranging from K-12 to Olympiad and advanced frontier mathematics, with 125 problems per language at each level.

<div align="center">
  <img src="_ASSETS/level.png" alt="logo" width="85%"/>
</div>



- 🌍 **Language Diversity:** Each problem in PolyMath is available in **18 parallel language versions**, encompassing **over 75% of the world’s native speakers** and major language families, ensuring diversity across both high-resource and low-resource languages.

<div align="center">
  <img src="_ASSETS/language.png" alt="logo" width="50%"/>
</div>

- 🧑‍🏫 **High-Quality Annotation:** Each problem translation is **calibrated by language experts**, avoiding direct use of LLM-generated outputs and ensuring precise term and logical clarity.

<div align="center">
  <img src="_ASSETS/human.png" alt="logo" width="90%"/>
</div>




---

## 📊 Main Results

The leaderboard is continuously updated! See https://qwen-polymath.github.io/#leaderboard

---

## 📄 Citation

If you use **PolyMath** in your research, please cite us:

```bibtex
@article{wang2025polymath,
  title={PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts},
  author={Yiming Wang and Pei Zhang and Jialong Tang and Haoran Wei and Baosong Yang and Rui Wang and Chenshu Sun and Feitong Sun and Jiran Zhang and Junxuan Wu and Qiqian Cang and Yichang Zhang and Fei Huang and Junyang Lin and Fei Huang and Jingren Zhou},
  journal={arXiv preprint arXiv:2504.18428},
  year={2025},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2504.18428}, 
}
```
