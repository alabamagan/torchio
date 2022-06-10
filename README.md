<!-- markdownlint-disable -->

<p align="center">
  <a href="http://torchio.rtfd.io/">
    <img src="https://raw.githubusercontent.com/fepegar/torchio/main/docs/source/favicon_io/for_readme_2000x462.png" alt="TorchIO logo">
  </a>
</p>
<!-- markdownlint-restore -->

> *Tools like TorchIO are a symptom of the maturation of medical AI research using deep learning techniques*.

Jack Clark, Policy Director
at [OpenAI](https://openai.com/) ([link](https://jack-clark.net/2020/03/17/)).

---

<!-- markdownlint-disable -->

<table align="center">
    <tr>
        <td align="left">
            <b>Package</b>
        </td>
        <td align="center">
            <a href="https://pypi.org/project/torchio/">
                <img src="https://img.shields.io/pypi/dm/torchio.svg?label=PyPI%20downloads&logo=python&logoColor=white" alt="PyPI downloads">
            </a>
            <a href="https://pypi.org/project/torchio/">
                <img src="https://img.shields.io/pypi/v/torchio?label=PyPI%20version&logo=python&logoColor=white" alt="PyPI version">
            </a>
            <a href="https://anaconda.org/conda-forge/torchio">
                <img src="https://img.shields.io/conda/v/conda-forge/torchio.svg?label=conda-forge&logo=conda-forge" alt="Conda version">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>CI</b>
        </td>
        <td align="center">
            <a href="https://github.com/fepegar/torchio/actions/workflows/tests.yml">
                <img src="https://github.com/fepegar/torchio/actions/workflows/tests.yml/badge.svg" alt="Tests status">
            </a>
            <a href="https://github.com/fepegar/torchio/actions/workflows/lint.yml">
                <img src="https://github.com/fepegar/torchio/actions/workflows/lint.yml/badge.svg" alt="Linting status">
            </a>
            <a href="https://torchio.rtfd.io/?badge=latest">
                <img src="https://img.shields.io/readthedocs/torchio?label=Docs&logo=Read%20the%20Docs" alt="Documentation status">
            </a>
            <a href="https://codecov.io/github/fepegar/torchio">
                <img src="https://codecov.io/gh/fepegar/torchio/branch/main/graphs/badge.svg" alt="Coverage status">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Code</b>
        </td>
        <td align="center">
            <a href="https://scrutinizer-ci.com/g/fepegar/torchio/?branch=main">
                <img src="https://img.shields.io/scrutinizer/g/fepegar/torchio.svg?label=Code%20quality&logo=scrutinizer" alt="Code quality">
            </a>
            <a href="https://codeclimate.com/github/fepegar/torchio/maintainability">
                <img src="https://api.codeclimate.com/v1/badges/518673e49a472dd5714d/maintainability" alt="Code maintainability">
            </a>
            <a href="https://github.com/pre-commit/pre-commit">
                <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Tutorials</b>
        </td>
        <td align="center">
            <a href="https://github.com/fepegar/torchio/blob/main/tutorials/README.md">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Google Colab">
            </a>
        </td>
    </tr>
    <tr>
        <td align="left">
            <b>Community</b>
        </td>
        <td align="center">
            <a href="https://join.slack.com/t/torchioworkspace/shared_invite/zt-exgpd5rm-BTpxg2MazwiiMDw7X9xMFg">
                <img src="https://img.shields.io/badge/TorchIO-Join%20on%20Slack-blueviolet?style=flat&logo=slack" alt="Slack">
            </a>
            <a href="https://twitter.com/TorchIOLib">
                <img src="https://img.shields.io/twitter/url/https/twitter.com/TorchIOLib.svg?style=social&label=Follow%20%40TorchIOLib" alt="Twitter">
            </a>
            <a href="https://twitter.com/TorchIO_commits">
                <img src="https://img.shields.io/twitter/url/https/twitter.com/TorchIO_commits.svg?style=social&label=Follow%20%40TorchIO_commits" alt="Twitter">
            </a>
            <a href="https://www.youtube.com/watch?v=UEUVSw5-M9M">
                <img src="https://img.shields.io/youtube/views/UEUVSw5-M9M?label=watch&style=social" alt="YouTube">
            </a>
            <a href="https://github.com/fepegar/torchio#contributors">
                <img src="https://img.shields.io/badge/Contributors-27-orange.svg" alt="Contributors">
            </a>
        </td>
    </tr>
</table>

# About this fork

This fork was made to cater for a number of missing functions that I used with the repo

# List of changes

---

* Fix the default value for masking_method in random_rescale 43fa2a2
* Add option to control which side (head, tail or equal) to pad when using the crop_or_pad filter f374a91
* Add an augmentation `RandomRescale` 279e3508
* Add more supports to the inference using the weighted sampler 9826bc73 a8ac0931
* Add `max` mode for aggregator to piece patches together. a8ac0931 1e1aa90b
