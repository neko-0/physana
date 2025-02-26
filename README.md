# Collinear W+jets Plotting Scripts

Python scripts for generating histograms from `ROOT.TTree` and making plots

## Installation

Setup python3 virtual environment via `source collinearw/envsetup.sh`.
Then install the package:

```bash
# for base
python -m pip install -e .
# for unfolding
python -m pip install -e .[unfolding]
# for complete
python -m pip install -e .[complete]
```

The `--no-cache-dir` is needed because of `pip` interactions with SLAC's `GPFS`, and this is set via `export PIP_NO_CACHE_DIR=1`.

For the next login, do `source setup_slac.sh` and you're good to go.

## Quick Start Guide

### Setting up with `ConfigMgr`

The `ConfigMgr` class is the top level container for most of the structures
(processes/regions/histograms), and other steering parameters. For example:

```python
from collinearw import ConfigMgr

config = ConfigMgr()
config.set_output_location("output") # optional

# to add process, regions, and histograms
config.add_process("wjets", treename="wjets_NoSys", filename=["f1.root", "f2.root"])
config.add_region("inclusive", selection="pt>500", weight="eventWeight")
config.add_observable("wpt", 100, 500, 3000)

config.prepare()
config.save("unfilled_config.pkl")
```

For more detailed example, see
[configs/main/build_histograms.py](configs/main/build_histograms.py).

----

### Filling Histograms

The `HistMaker` class is responsible to iterating through the ROOT
files and ttrees, and filling histograms within the instance of `ConfigMgr`
In most of cases, the filling step can be done via the `run_HistMaker` interface:

```python
from collinearw import ConfigMgr, run_HistMaker

# open unfilled config file
config = ConfigMgr.open("unfilled_config.pkl")

# filling config
config = run_HistMaker.run_HistMaker(config)

config.save("filled_config.pkl")
```

### Retrieving Histogram Objects

Histogram objects can be retrieved via the following:

```python
config = ConfigMgr.open("filled_config.pkl") # open the filled ConfigMgr

# full path approach
h_pt = config.get("wjets//nominal//inclusive//pt")

# 'get()' chain approach
h_pt = config.get("wjets").get(None).get("inclusive").get("wpt")
```

### Adding New Phase-space and Histogram to ConfigMgr

You can store new `Region` (phase-space) and histograms into an
existing `ConfigMgr` object:

```python
import numpy as np
from collinearw import ConfigMgr, Region, Histogram

config = ConfigMgr.open("filled_config.pkl")

# create new region and histogram
new_region = Region("collinear", weights="eventWeight", selection="dR < 2.6")
njet_h = Histogram("jet multiplicity", 10, 0, 10, observable="nJet30")

# the histogram is not fill here, so you can populated by yourself
njet_h.from_array(np.random.randint(0, 10, 1000))

# add histogram to region
new_region.add_histogram(njet_h)

# add region to existing process
config.get("wjets//nominal").add_region(new_region)

```

## CMD line utilites

A set of command line tools to handle `ConfigMgr` pickled objects.

### Print out content

You can print out the content of a configuration manager file using

```bash
collinearw utility browse my_config.pkl
```

----

### Merging and intersection

The mergining and intersection tools can be used via cmd line interface.
for example, merge multiple `ConfigMgr` without duplication and addition can be done via intersection:

```bash
collinearw utility intersection --output output.pkl --input "input1.pkl,input2.pkl"
collinearw utility intersection --output output.pkl --input "inputs/*pkl"

```
### Merging migration phasespace

To included the phasespace migration effect from another `ConfigMgr` object, use

```bash
collinearw utility merge-migration --output merged.pkl --nominal nominal.pkl --migratein migration.pkl
```

It assumes both `ConfigMgr` objects contain the same `process/regions/histogram` naming.

### Extracting nominal processes

Extract all of the nominal processes from a `ConfigMgr` objects that contains systematics

```bash
collinearw extract-nominal --input input.pkl --output nonminal_only.pkl
```

### Filtering process sets

Process sets can be filtered by

```bash
collinearw filter-process --input input.pkl --output filtered.pkl --exclude "ttbar,zjets" # removing 'ttbar' & 'zjets'
collinearw filter-process --input input.pkl --output filtered.pkl --include "wjets" # only select 'wjets
```

### Merging replica runs

Merging multiple `ConfigMgr` objects that were genereated from `bootsrap` routine.

```bash
collinearw merge-replica --output merged.pkl --configs "replica_1.pkl,replica_2.pkl"
```

## Old (require updates)

#### Generate histogram from ROOT files.

After writing your configMgr, you can start making histograms.

For example, try:

```
collinearw histmaker run --config configs/template_1.py --oname "my_first_example"
```

where:
 * `--config` is the path to your python config file
 * `--oname` is the output name

#### Generate histogram with external weights.

You can add an external weights in the HistMaker process. For example, this can be used in estimating fake background.

the command is:
```
collinearw histmaker weight-gen --config configs/ABCD_FakeBG.py --oname "ABCD_Study_FKBG" --weight-obs "met" --weight-file ABCD_FakeBG_lep1Signal_vs_met_ABCD.json
```

where:
  * `--weight-obs` is the observable of the weight distribution
  * `--weight-file` is the file contains the weight distribution

Currently it only supports the weight distribution generated from `conllinearw.HistManipulate.ABCD` method.

#### TODO
need to fix when the config file is not present.

### HistManipulate

#### ABCD Fake Background

example
```
collinearw histmanipulate abcd --config /nfs/slac/atlas/fs1/d/yuzhan/new_production_2020_Jan/ABCD_v2/simple_ABCD_study_BGSub.pkl --process data_sub__wjets__ttbar__singletop__diboson_ --xline 25 --region "collinear_50_All_MT40H_e" --tag "_MT40H_e" --oname "ABCD_result_MT40H_e"
```

getting the fake for other observable, for example, run

```
collinearw histmanipulate fake --config simple_fake.pkl --process data_sub__wjets__ttbar__singletop__diboson_ --abcd_r collinear_50_All_MT40L_mu --obs lep1Signal_vs_met --d_r collinear_50_D_MT40L_mu --a_r collinear_50_Sig_MT40L_mu --oname "fake"

```

#### Background or process subtraction

You can subtract process or background through this command line:
```
ollinearw histmanipulate subprocess --config simple_abcd_lep1Signal.pkl --process data --backgrounds "wjets,ttbar,singletop,diboson" --oname "abcd_lep1_bgsub" --bg_fluc "5,10,50,100,200,500" --proc_fluc 1
```

where:
  * `--config` is the path configuration manager file.
  * `--backgrounds` is the name of the background processes, each process is separated by comma.
  * `--process` is name the main process that you want to subtract.
  * `--bg_fluc` is percentage fluctuation in background.
  * `--proc_fluc` is the on/off switch for poisson fluctuation in the main process
  * `--oname` is the output name.

### PlotMaker

To Be Documented

#### Making stack plots
You can make stack plots using the command line. For example:
```
collinearw plotmaker plot-stack --config simple_ABCD_study.pkl --output_tag "test" --data data --mc "diboson,ttbar,singletop,wjets" --region collinear_50_All_MT40H_e --observable "jet1Pt,lep1Pt,met" --odir "./"
```

## What giordon does

```
collinearw histmaker run --config configs/lxplus_AB21298_v1.py --oname output
collinearw histmanipulate root --config user_data/Wj_AB212108_v1a_FixedCutBEff60/obj_output_HistMaker_.pkl
collinearw plotmaker run --config user_data/Wj_AB212108_v1a_FixedCutBEff60/obj_output_HistMaker_.pkl
```


## Unfolding

See [RooUnfold documentation](https://gitlab.cern.ch/RooUnfold/RooUnfold) for details but... to set it up, `git submodule update --init --recursive` and then `mkdir build; cd build; cmake ../RooUnfold; make -j4`.

## Grabbing data files

```
git lfs pull --include "data/"
```

## Special Cases

### Un-merged files from SusySkimAna

If the merge step `run_merge` from `SusySkimAna` takes too long, there is an option to calculate the `eventWeight` on-the-fly with user specified integrated luminosity, process cross section, and sum of event weights.

To use this option, we need to include the following:

```python
from collinearw import ConfigMgr, XSecSumEvtW

config = ConfigMgr()
config.xsec_sumw = XSecSumEvtW("sample.json")
```

where the `sample.json` has the following format:

```json
{
    "361600": {
        "kinematic_1": {
            "xsec": 10.635,
            "AllExecutedEvents_sumOfEventWeights": 157097000.0
        },
        "weight_1": {
            "xsec": 10.635,
            "AllExecutedEvents_sumOfEventWeights": 157097000.0
        },
        "weight_2": {
            "xsec": 10.635,
            "AllExecutedEvents_sumOfEventWeights": 157097000.0
        }
    }
}
```

The `XSecSumEvtW` class relies on the path name of the input ROOT N-tuple files. This class will take path name and tokenize the information into `dsid`, `process`, `user` and etc. Users can specify their own rules on how to parse the file path. The rules need to be regular expression, and the default parsing rule is

```regexp
"(.*)(user).(.\w*).(.\w*).(.\d*).(.*).(CollinearW.SMA.v1)_(.*)_(t2_tree.root)"
```

the default tokenize grouping rules is the following

```python
XSecSumEvtW.token_groups_rule = {
  "path": 1,
  "prefix": 2,
  "user": 3,
  "process": 4,
  "dsid": 5,
  "ptag": 6,
  "user-tag": 7,
  "syst": 8,
  "suffix": 9,
}
```

Please stick with the same keys as defined above since the lookup and calculation of `eventWeight` will rely on the keys.

Here is an example of standalone usage:

```python
from collinearw import XSecSumEvtW

xsec = XSecSumEvtW("sample.json")
xsec.match("/gpfs/slac/atlas/fs1/d/mgignac/analysis/mc16a/CollinearW.SMA.v1/trees/user.mgignac.singletop.410659.e6671_s3126_r9364_p4512_CollinearW.SMA.v1_kinematic_1_t2_tree.root")

print(xsec['user']) # mgignac
print(xsec['process']) # singletop
print(xsec['dsid']) # 410659
print(xsec['syst']) # kinematic_1
print(xsec['user-tag']) # CollinearW.SMA.v1
print(xsec.get_xsec_sumw()) # lumi*xsec/sumOfWeights = 4.787780160984935e-08
```

Here is an example usage with `ConfigMgr`.

```python
from collinearw import ConfigMgr, XSecSumEvtW

config = ConfigMgr.open("prepared_config.pkl") # previously prepared ConfigMgr object
config.default_wlist = [
    # "genWeight", # no longer need genWeight if XSecSumEvtW is specified.
    "eventWeight",
    "bTagWeight",
    "pileupWeight",
    "triggerWeight",
    "jvtWeight",
    "leptonWeight",
]

xsec_sumw = XSecSumEvtW()

# for MC campaign sensitive runs
xsec_sumw.campaign_sensitive = True
xsec_sumw.set_campaign_files(
  {
    "a" : "sampleNorm_2_mc16a.json",
    "d" : "sampleNorm_mc16d.json",
    "e" : "sampleNorm_mc16e.json",
  }
)
xsec_sumw.set_campaign_lumi(
    {
        "a" : 36646.74,
        "d" : 44630.60,
        "e" : 58791.60,
    }
)

xsec_sumw.nominal_token = "weight_1"
xsec_sumw.do_check_process = True

# name mapping to process name identified in the file path.
xsec_sumw.check_map = {
    "wjets_tau" : "wjets_FxFx",
    "vgamma" : "vgamma_sherpa2211",
    "ttbar" : "ttbar_Sherpa2212",
    "ttbar_TF_1_UP" : "ttbar_Sherpa2212",
    "ttbar_TF_1_DOWN" : "ttbar_Sherpa2212",
    "zjets_2211_TF_1_UP" : "zjets_2211",
    "zjets_2211_TF_1_DOWN" : "zjets_2211",
    "dijets_CR_Def_Max" : "dijets", 
    "dijets_CR_Def_Min" : "dijets", 
}

# pass in the saved JSON file location. NOT the XSecSumEvtW object
setting.xsec_sumw = xsec_sumw.save("sumw_file.json")

```