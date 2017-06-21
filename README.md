# DIANA/HEP Fellowship for Matthew Feickert (Project Name is work in progress: DFGMark?)

[![license](https://img.shields.io/github/license/matthewfeickert/fellowship-project.svg)]() [![nbviewer](https://img.shields.io/badge/view%20on-nbviewer-brightgreen.svg)](http://nbviewer.jupyter.org/github/matthewfeickert/fellowship-project/tree/master/Notebooks/) [![SWAN](http://swanserver.web.cern.ch/swanserver/images/badge_swan_white_150.png)](https://swan002.cern.ch/?projurl=https://github.com/matthewfeickert/fellowship-project.git)

There are examples of how to use the tool in the form of Jupyter notebooks in the `Notebooks` directory.

## Project Timeline

- May 8th through May 21st

  - [x] Create a GitHub repository for the project.
  - [ ] Design a template for binned models that is parametrized in terms of number of events, number of bins, number of channels, number of signal/background components, and number of parameters of interest and nuisance parameters.
  - [x] Begin documentation of the template model on the GitHub repository.
    - See `Notebooks` directory

- May 22nd through June 4th

  - [ ] Establish a precise mathematical formulation that is implementation-independent based on the `HistFactory` schema.
  - [x] Begin study of data flow graph frameworks or of probabilistic programming frameworks.
    - Studying `TensorFlow` and `Edward`
  - [ ] Implement the template for the benchmark models with a `HistFactory` script.

- June 5th through June 18th

  - [x] Conclude from study of data flow graph frameworks or of probabilistic programming frameworks which framework to pursue.
    - Using [`Edward`](http://edwardlib.org/)
  - [ ] Begin implementation of the benchmark models in the selected framework.
  - [ ] Begin to write technical report.

- **June 19th through July 2nd**

  - [ ] Finish implementation of the benchmark models in the selected framework.
  - [ ] Apply benchmarks evaluating framework data and model parallelism.

- July 3rd through July 16th

  - [ ] Finish technical report.
  - [ ] Create tutorial on data flow graphs targeted for physicists.

- July 17th through August 8th

  - [ ] Upstream software contributions to address the identified limitations (if any).
  - [ ] Development of/Contribution to a probabilistic framework (if time permits).

## Authors

Primary Author: [Matthew Feickert](http://www.matthewfeickert.com/)

## Acknowledgments

This work is supported by the [DIANA/HEP project](http://diana-hep.org/), which if funded solely by the National Science Foundation ([NSF ACI-1450310](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1450310)).

- Project mentors and advisors:

  - [Gilles Louppe](https://glouppe.github.io/): Project mentor and collaborator
  - [Vince Croft](https://www.nikhef.nl/~vcroft/): Project mentor and collaborator
  - [Kyle Cranmer](http://as.nyu.edu/faculty/kyle-s-cranmer.html.html): Project advisor
  - [Stephen Sekula](http://www.physics.smu.edu/sekula/): Matthew's research advisor

- Many thanks to [Lukas Heinrich](https://github.com/lukasheinrich) for insightful discussions.

- All badges made by [shields.io](http://shields.io/)
