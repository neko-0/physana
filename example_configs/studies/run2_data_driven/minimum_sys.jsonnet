# list of systematic definition for collinearw ConfigMgr

#####
# Add a atrritbute to specify correlation
# of muon and electron 
#####

# function to append 'up' and 'down'
local OneUpDown(name) = [name+"__1up", name+"__1down"];

local LHE(name) = "LHE3Weight_" + name;

local add_process_prefix(name, list) = [name + "_" + x for x in list];

# correlate JET_NP, JET_JER, MET

local JET_NP_List = [
  "JET_GroupedNP_1",
  "JET_GroupedNP_2",
  "JET_GroupedNP_3",
]; # correlate

local JET_JER_List = [
  "JET_JER_DataVsMC_MC16",
  "JET_JER_EffectiveNP_1",
  "JET_JER_EffectiveNP_2",
  "JET_JER_EffectiveNP_3",
  "JET_JER_EffectiveNP_4",
  "JET_JER_EffectiveNP_5",
  "JET_JER_EffectiveNP_6",
  "JET_JER_EffectiveNP_7restTerm",
]; # correlate

local MET_SoftTrk_List = [
  "MET_SoftTrk_ResoPara",
  "MET_SoftTrk_ResoPerp",
  "MET_SoftTrk_ScaleDown",
  "MET_SoftTrk_ScaleUp",
]; # correlate

local JET_EtaIntercalibration_NonClosure_List = [
  "JET_EtaIntercalibration_NonClosure_2018data",
  "JET_EtaIntercalibration_NonClosure_highE",
  "JET_EtaIntercalibration_NonClosure_negEta",
  "JET_EtaIntercalibration_NonClosure_posEta",
]; # correlate

local EG_List = [
  "EG_RESOLUTION_ALL",
  "EG_SCALE_AF2",
  "EG_SCALE_ALL",
]; # de-correlate

local MUON_List = [
  "MUON_ID",
  "MUON_MS",
  "MUON_SAGITTA_RESBIAS",
  "MUON_SAGITTA_RHO",
  "MUON_SCALE",
]; # de-correlate

local bTagWeight_List = [
  "bTagWeight_FT_EFF_B_systematics",
  "bTagWeight_FT_EFF_C_systematics",
  "bTagWeight_FT_EFF_Light_systematics",
  "bTagWeight_FT_EFF_extrapolation",
  "bTagWeight_FT_EFF_extrapolation_from_charm",
]; # correlate b-tag ???

local jvtWeight_List = [
  "jvtWeight_JET_JvtEfficiency",
]; # correlate

local leptonWeight_EL_List = [
  "leptonWeight_EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR",
  "leptonWeight_EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR",
  "leptonWeight_EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR",
]; # de-correlate

local leptonWeight_MU_List = [
  "leptonWeight_MUON_EFF_ISO_STAT",
  "leptonWeight_MUON_EFF_RECO_STAT",
  "leptonWeight_MUON_EFF_RECO_SYS",
  "leptonWeight_MUON_EFF_TTVA_STAT",
  "leptonWeight_MUON_EFF_TTVA_SYS",
]; # correlate

# trigger weights
local trigweight_EL_list = [
  "trigWeight_EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR",
]; # de-correlate

local trigWeight_MU_list = [
  "trigWeight_MUON_EFF_TrigStatUncertainty",
  "trigWeight_MUON_EFF_TrigSystUncertainty",
]; # de-correlate


# V+jets theory

local Vjets2211_MUR_MUF_Scale = [
  "MUR0p5_MUF1_PDF303200_PSMUR0p5_PSMUF1",
  "MUR0p5_MUF0p5_PDF303200_PSMUR0p5_PSMUF0p5",
  "MUR1_MUF0p5_PDF303200_PSMUR1_PSMUF0p5",
  "MUR1_MUF2_PDF303200_PSMUR1_PSMUF2",
  "MUR2_MUF1_PDF303200_PSMUR2_PSMUF1",
  "MUR2_MUF2_PDF303200_PSMUR2_PSMUF2",
]; # correlate

local Vjets_NNPDF30nnlo_hessian_PDF = [
  "MUR1_MUF1_PDF303%03d"%i for i in std.range(201,300)
]; # correlate

local Vjets_FxFx_MUR_MUF_Scale = [
  "MUR0p5_MUF0p5_PDF325100",
  "MUR0p5_MUF1p0_PDF325100",
  "MUR0p5_MUF2p0_PDF325100",
  "MUR1p0_MUF0p5_PDF325100",
  "MUR1p0_MUF2p0_PDF325100",
  "MUR2p0_MUF0p5_PDF325100",
  "MUR2p0_MUF1p0_PDF325100",
  "MUR2p0_MUF2p0_PDF325100",
]; # correlate

local Vjets_FxFx_PDF = [
  "MUR1p0_MUF1p0_PDF325%03d"%i for i in std.range(101,200)
]; # correlate

local Vjets221_PDF = [
  "MUR1_MUF1_PDF261%03d"%i for i in std.range(1,100)
]; # correlate

local Vjets221_Scale = [
  "MUR0p5_MUF0p5_PDF261000",
  "MUR0p5_MUF1_PDF261000",
  "MUR1_MUF0p5_PDF261000",
  "MUR1_MUF2_PDF261000",
  "MUR2_MUF1_PDF261000",
  "MUR2_MUF2_PDF261000",
]; # correlate

# sinlgetop theory

local singletop_A14_tunning = [
  "Var3cDown",
  "Var3cUp",
]; # correlate

local singletop_PDF = [
  "PDFset260%03d"%i for i in std.range(1,100)
];# correlate

local singletop_ren_fac_scale = [
  "muR0p50muF0p50",
  "muR0p50muF1p00",
  "muR0p50muF2p00",
  "muR1p00muF2p00",
  "muR1p00muF0p50",
  "muR2p00muF0p50",
  "muR2p00muF1p00",
  "muR2p00muF2p00",
];# correlate

# ttbar theory

local ttbar_matrix_element = ["aMcAtNloPy8_NoSys"];# correlate

local ttbar_parton_shower = ["PowhegHerwig713_NoSys"];# correlate

local ttbar_ren_fac_scale = [
  "muR0p5muF2p0",
  "muR0p5muF1p0",
  "muR0p5muF0p5",
  "muR1p0muF2p0",
  "muR1p0muF0p5",
  "muR2p0muF0p5",
  "muR2p0muF1p0",
  "muR2p0muF2p0",
];# correlate

local ttbar_ISR_scale = [
  "Var3cDown",
  "Var3cUp",
];# correlate

local ttbar_FSR_scale = [
  "isrmuRfac1p0_fsrmuRfac2p0",
  "isrmuRfac1p0_fsrmuRfac0p5",
];# correlate

local ttbar_NNPDF30_PDF = [
  "PDFset260%03d"%i for i in std.range(1,100)
];# correlate

# diboson theory

local Flat10Percent = ["1.10", "0.9"];# correlate

local diboson222_MUR_MUF_Scale = [
  "MUR0p5_MUF0p5_PDF261000",
  "MUR0p5_MUF1_PDF261000",
  "MUR1_MUF0p5_PDF261000",
  "MUR1_MUF2_PDF261000",
  "MUR2_MUF1_PDF261000",
  "MUR2_MUF2_PDF261000",
];# correlate

local diboson_NNPDF30_PDF = [
  "MUR1_MUF1_PDF261%03d"%i for i in std.range(1,100)
];# correlate

local diboson_powheg_MUR_MUF_Scale = [
  "diboson_powheg_muR1p0muF2p0",
  "diboson_powheg_muR1p0muF0p5",
  "diboson_powheg_muR2p0muF1p0",
  "diboson_powheg_muR2p0muF2p0",
  "diboson_powheg_muR2p0muF0p5",
  "diboson_powheg_muR0p5muF1p0",
  "diboson_powheg_muR0p5muF2p0",
  "diboson_powheg_muR0p5muF0p5p",
];# correlate

local diboson_powheg_PDF = [
  "pdfset11%03d"%i for i in std.range(1,52)
];# correlate

# dijets Pythia8 theory systematics
local Dijets_FlatVariation = ["1.15", "0.85"];

local dijets_pythia8_A14_tunning = [
  "Var3cDown",
  "Var3cUp",
];# correlate

local dijets_pythia8_FSR_scale = [
  "isrmuRfac1p0_fsrmuRfac0p5",
  "isrmuRfac1p0_fsrmuRfac0p625",
  "isrmuRfac1p0_fsrmuRfac0p75",
  "isrmuRfac1p0_fsrmuRfac0p875",
  "isrmuRfac1p0_fsrmuRfac1p25",
  "isrmuRfac1p0_fsrmuRfac1p5",
  "isrmuRfac1p0_fsrmuRfac1p75",
  "isrmuRfac1p0_fsrmuRfac2p0",
];# correlate

local dijets_pythia8_ISR_scale = [
  "isrmuRfac0p5_fsrmuRfac1p0",
  "isrmuRfac0p625_fsrmuRfac1p0",
  "isrmuRfac0p75_fsrmuRfac1p0",
  "isrmuRfac0p875_fsrmuRfac1p0",
  "isrmuRfac1p25_fsrmuRfac1p0",
  "isrmuRfac1p5_fsrmuRfac1p0",
  "isrmuRfac1p75_fsrmuRfac1p0",
  "isrmuRfac2p0_fsrmuRfac1p0",
];# correlate

/* local dijets_pythia_both_ISR_FSR_scale = [
  "isrmuRfac0p5_fsrmuRfac0p5",
  "isrmuRfac2p0_fsrmuRfac2p0",
] */

# define and settting systematics

{
  "tree_base" : [
    {
      "name" : x,
      "tlist" : OneUpDown(x),
      "handle" : "up_down",
      "sys_type" : "experiment",
    } for x in JET_NP_List
  ]

  +
  [
    {
      "name" : x,
      "tlist" : OneUpDown(x),
      "handle" : "up_down",
      "sys_type" : "experiment",
    } for x in JET_JER_List
  ]

  +
  [
    {
      "name" : x,
      "tlist" : OneUpDown(x),
      "handle" : "up_down",
      "sys_type" : "experiment",
    } for x in JET_EtaIntercalibration_NonClosure_List
  ]

  +
  [
    {
      "name" : x,
      "tlist" : [x],
      "handle" : "up_down",
      "sys_type" : "experiment",
    } for x in MET_SoftTrk_List
  ]

  +
  [
    {
      "name" : x,
      "tlist" : OneUpDown(x),
      "handle" : "up_down",
      "sys_type" : "experiment",
    } for x in EG_List
  ]

  +
  [
    {
      "name" : x,
      "tlist" : OneUpDown(x),
      "handle" : "up_down",
      "sys_type" : "experiment",
    } for x in MUON_List
  ]

  +
  [
    {
      "name" : "ttbar_matrix_element",
      "tlist" : ttbar_matrix_element,
      "handle" : "up_down",
      "sys_type" : "theory",
    }
  ]

  +
  [
    {
      "name" : "ttbar_parton_shower",
      "tlist" : ttbar_parton_shower,
      "handle" : "up_down",
      "sys_type" : "theory",
    }
  ],

  "weight_base" : [
      {
        "name" : x,
        "wlist" : [ y+"/bTagWeight" for y in OneUpDown(x) ],
        "handle" : "up_down",
        "sys_type" : "experiment",
      } for x in bTagWeight_List
    ]

    +
    [
      {
        "name" : x,
        "wlist" : [ y+"/jvtWeight" for y in OneUpDown(x) ],
        "handle" : "up_down",
        "sys_type" : "experiment",
      } for x in jvtWeight_List
    ]

    +
    [
      {
        "name" : x,
        "wlist" : [ y+"/leptonWeight" for y in OneUpDown(x) ],
        "handle" : "up_down",
        "sys_type" : "experiment",
      } for x in leptonWeight_EL_List + leptonWeight_MU_List
    ]

    +
    [
      {
        "name" : name + "_MUR_MUF_Scale",
        "wlist" : [LHE(x) for x in Vjets2211_MUR_MUF_Scale],
        "handle" : "min_max",
        "sys_type" : "theory",
      } for name in ["wjets_2211", "zjets_2211"]
    ]

    +
    [
      {
        "name" : name + "_MUR_MUF_Scale",
        "wlist" : [LHE(x) for x in Vjets_FxFx_MUR_MUF_Scale],
        "handle" : "min_max",
        "sys_type" : "theory",
      } for name in ["wjets_FxFx"]
    ]

    +
    [
      {
        "name" : name + "_MUR_MUF_Scale",
        "wlist" : [LHE(x) for x in Vjets221_Scale],
        "handle" : "min_max",
        "sys_type" : "theory",
      } for name in ["wjets"]
    ]

    +
    [
      {
        "name" : name + "_NNPDF30nnlo_hessian",
        "wlist" : [LHE(x) for x in Vjets_NNPDF30nnlo_hessian_PDF],
        "handle" : "hessian",
        "sys_type" : "theory",
      } for name in ["wjets_2211", "zjets_2211"]
    ]

    +
    [
      {
        "name" : name + "_PDF",
        "wlist" : [LHE(x) for x in Vjets_FxFx_PDF],
        "handle" : "std",
        "sys_type" : "theory",
      } for name in ["wjets_FxFx"]
    ]

    +
    [
      {
        "name" : name + "_PDF",
        "wlist" : [LHE(x) for x in Vjets221_PDF],
        "handle" : "std",
        "sys_type" : "theory",
      } for name in ["wjets"]
    ]

    +
    [
      {
        "name" : name + "_PDF",
        "wlist" : [LHE(x) for x in singletop_PDF],
        "handle" : "std",
        "sys_type" : "theory"
      } for name in ["singletop"]
    ]

    +
    [
      {
        "name" : name + "_A14_Tunning",
        "wlist" : [LHE(x) for x in singletop_A14_tunning],
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["singletop"]
    ]

    +
    [
      {
        "name" : name + "_ren_fac_scale",
        "wlist" : [LHE(x) for x in singletop_ren_fac_scale],
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["singletop"]
    ]

    +
    [
      {
        "name" : name + "_ren_fac_scale",
        "wlist" : [LHE(x) for x in ttbar_ren_fac_scale ],
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["ttbar"]
    ]

    +
    [
      {
        "name" : name + "_ISR_scale",
        "wlist" : [LHE(x) for x in ttbar_ISR_scale ],
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["ttbar"]
    ]

    +
    [
      {
        "name" : name + "_FSR_scale",
        "wlist" : [LHE(x) for x in ttbar_FSR_scale ],
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["ttbar"]
    ]

    +
    [
      {
        "name" : name + "_NNPDF30_PDF",
        "wlist" : [LHE(x) for x in ttbar_NNPDF30_PDF ],
        "handle" : "std",
        "sys_type" : "theory"
      } for name in ["ttbar"]
    ]

    #+
    #[
    #  {
    #    "name" : name + "_Flat10Percent",
    #    "wlist" : Flat10Percent ,
    #    "handle" : "min_max",
    #    "sys_type" : "theory"
    #  } for name in ["diboson"]
    #]

    +
    [
      {
        "name" : name + "_A14_Tunning",
        "wlist" : [LHE(x) for x in dijets_pythia8_A14_tunning ],
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["dijets"]
    ]

    +
    [
      {
        "name" : name + "_FSR_scale",
        "wlist" : [LHE(x) for x in dijets_pythia8_FSR_scale ],
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["dijets"]
    ]

    +
    [
      {
        "name" : name + "_ISR_scale",
        "wlist" : [LHE(x) for x in dijets_pythia8_ISR_scale ],
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["dijets"]
    ]

    +
    [
      {
        "name" : name + "_NNPDF30_PDF",
        "wlist" : [LHE(x) for x in diboson_NNPDF30_PDF ] ,
        "handle" : "stdev",
        "sys_type" : "theory"
      } for name in ["diboson"]
    ]

    +
    [
      {
        "name" : name + "_MUR_MUF_Scale",
        "wlist" : [LHE(x) for x in diboson222_MUR_MUF_Scale ] ,
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["diboson"]
    ]

    +
    [
      {
        "name" : name + "_MUR_MUF_Scale",
        "wlist" : [LHE(x) for x in diboson_powheg_MUR_MUF_Scale ] ,
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["diboson_powheg"]
    ]

    +
    [
      {
        "name" : name + "_PDF",
        "wlist" : [LHE(x) for x in diboson_powheg_PDF ] ,
        "handle" : "std",
        "sys_type" : "theory"
      } for name in ["diboson_powheg"]
    ]

    +
    [
      {
        "name" : x,
        "wlist" : [ y+"/triggerWeight" for y in OneUpDown(x) ],
        "handle" : "up_down",
        "sys_type" : "experiment",
      } for x in trigweight_EL_list + trigWeight_MU_list
    ]
    ,

  "set_systematics" : {
    "wjets_2211" : JET_EtaIntercalibration_NonClosure_List + trigweight_EL_list + trigWeight_MU_list + leptonWeight_EL_List + leptonWeight_MU_List + jvtWeight_List + MUON_List + EG_List + MET_SoftTrk_List + JET_JER_List + JET_NP_List + bTagWeight_List + add_process_prefix("wjets_2211", ["MUR_MUF_Scale", "NNPDF30nnlo_hessian"]),
    "zjets_2211" : JET_EtaIntercalibration_NonClosure_List + trigweight_EL_list + trigWeight_MU_list + leptonWeight_EL_List + leptonWeight_MU_List + jvtWeight_List + MUON_List + EG_List + MET_SoftTrk_List + JET_JER_List + JET_NP_List + bTagWeight_List + add_process_prefix("zjets_2211", ["MUR_MUF_Scale", "NNPDF30nnlo_hessian"]),
    "ttbar" : JET_EtaIntercalibration_NonClosure_List + trigweight_EL_list + trigWeight_MU_list + leptonWeight_EL_List + leptonWeight_MU_List + jvtWeight_List + MUON_List + EG_List + MET_SoftTrk_List + JET_JER_List + JET_NP_List + bTagWeight_List + ["ttbar_matrix_element", "ttbar_parton_shower"] + add_process_prefix("ttbar", ["ren_fac_scale","ISR_scale","FSR_scale","NNPDF30_PDF"]),
    "dijets" : JET_EtaIntercalibration_NonClosure_List + trigweight_EL_list + trigWeight_MU_list + leptonWeight_EL_List + leptonWeight_MU_List + jvtWeight_List + MUON_List + EG_List + MET_SoftTrk_List + JET_JER_List + JET_NP_List + bTagWeight_List + add_process_prefix("dijets", ["ISR_scale", "FSR_scale", "A14_Tunning"]),
    "diboson_powheg" : JET_EtaIntercalibration_NonClosure_List + trigweight_EL_list + trigWeight_MU_list + leptonWeight_EL_List + leptonWeight_MU_List + jvtWeight_List + MUON_List + EG_List + MET_SoftTrk_List + JET_JER_List + JET_NP_List + bTagWeight_List + add_process_prefix("diboson_powheg", ["PDF", "MUR_MUF_Scale"]),
    "singletop" : JET_EtaIntercalibration_NonClosure_List + trigweight_EL_list + trigWeight_MU_list + leptonWeight_EL_List + leptonWeight_MU_List + jvtWeight_List + MUON_List + EG_List + MET_SoftTrk_List + JET_JER_List + JET_NP_List + bTagWeight_List + add_process_prefix("singletop", ["ren_fac_scale", "PDF", "A14_Tunning"]),
    #"wjets_FxFx" : JET_EtaIntercalibration_NonClosure_List + trigweight_EL_list + trigWeight_MU_list + leptonWeight_EL_List + leptonWeight_MU_List + jvtWeight_List + MUON_List + EG_List + MET_SoftTrk_List + JET_JER_List + JET_NP_List + bTagWeight_List + add_process_prefix("wjets_FxFx", ["MUR_MUF_Scale", "PDF"]),
    "wjets_FxFx" : add_process_prefix("wjets_FxFx", ["MUR_MUF_Scale", "PDF"]),
    "wjets" : add_process_prefix("wjets", ["MUR_MUF_Scale", "PDF"]),
    #"diboson" : JET_EtaIntercalibration_NonClosure_List + trigweight_EL_list + trigWeight_MU_list + leptonWeight_EL_List + leptonWeight_MU_List + jvtWeight_List + MUON_List + EG_List + MET_SoftTrk_List + JET_JER_List + JET_NP_List + bTagWeight_List + add_process_prefix("diboson", ["MUR_MUF_Scale", "NNPDF30_PDF"]),
    #"wjets_2211" : add_process_prefix("wjets_2211", ["MUR_MUF_Scale", "NNPDF30nnlo_hessian"]),
    #"wjets_2211" : JET_JER_List,
    #"zjets_2211" :JET_JER_List,
    #"diboson" : JET_JER_List,
    #"ttbar" :JET_JER_List,
    #"singletop" :JET_JER_List,
  },

  "branch_rename": {
    "LHE3Weight_muR1p000000E+00muF2p000000E+00" : "LHE3Weight_diboson_powheg_muR1p0muF2p0",
    "LHE3Weight_muR1p000000E+00muF5p000000E-01" : "LHE3Weight_diboson_powheg_muR1p0muF0p5",
    "LHE3Weight_muR2p000000E+00muF1p000000E+00" : "LHE3Weight_diboson_powheg_muR2p0muF1p0",
    "LHE3Weight_muR2p000000E+00muF2p000000E+00" : "LHE3Weight_diboson_powheg_muR2p0muF2p0",
    "LHE3Weight_muR2p000000E+00muF5p000000E-01" : "LHE3Weight_diboson_powheg_muR2p0muF0p5",
    "LHE3Weight_muR5p000000E-01muF1p000000E+00" : "LHE3Weight_diboson_powheg_muR0p5muF1p0",
    "LHE3Weight_muR5p000000E-01muF2p000000E+00" : "LHE3Weight_diboson_powheg_muR0p5muF2p0",
    "LHE3Weight_muR5p000000E-01muF5p000000E-01" : "LHE3Weight_diboson_powheg_muR0p5muF0p5p",
  }
}
