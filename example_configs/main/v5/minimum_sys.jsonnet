# list of systematic definition for collinearw ConfigMgr

# function to append 'up' and 'down'
local OneUpDown(name) = [name+"__1up", name+"__1down"];

local LHE(name) = "LHE3Weight_" + name;

local add_process_prefix(name, list) = [name + "_" + x for x in list];

local JET_NP_List = [
  "JET_GroupedNP_1",
  "JET_GroupedNP_2",
  "JET_GroupedNP_3",
];

local JET_JER_List = [
  "JET_JER_DataVsMC_MC16",
  "JET_JER_EffectiveNP_1",
  "JET_JER_EffectiveNP_2",
  "JET_JER_EffectiveNP_3",
  "JET_JER_EffectiveNP_4",
  "JET_JER_EffectiveNP_5",
  "JET_JER_EffectiveNP_6",
  "JET_JER_EffectiveNP_7restTerm",
];

local MET_SoftTrk_List = [
  "MET_SoftTrk_ResoPara",
  "MET_SoftTrk_ResoPerp",
  "MET_SoftTrk_ScaleDown",
  "MET_SoftTrk_ScaleUp",
];

local JET_EtaIntercalibration_NonClosure_List = [
  "JET_EtaIntercalibration_NonClosure_2018data",
  "JET_EtaIntercalibration_NonClosure_highE",
  "JET_EtaIntercalibration_NonClosure_negEta",
  "JET_EtaIntercalibration_NonClosure_posEta",
];

local EG_List = [
  "EG_RESOLUTION_ALL",
  "EG_SCALE_AF2",
  "EG_SCALE_ALL",
];

local MUON_List = [
  "MUON_ID",
  "MUON_MS",
  "MUON_SAGITTA_RESBIAS",
  "MUON_SAGITTA_RHO",
  "MUON_SCALE",
];


local bTagWeight_List = [
  "bTagWeight_FT_EFF_B_systematics",
  "bTagWeight_FT_EFF_C_systematics",
  "bTagWeight_FT_EFF_Light_systematics",
  "bTagWeight_FT_EFF_extrapolation",
  "bTagWeight_FT_EFF_extrapolation_from_charm",
];

local jvtWeight_List = [
  "jvtWeight_JET_JvtEfficiency",
];

local leptonWeight_EL_List = [
  "leptonWeight_EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR",
  "leptonWeight_EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR",
  "leptonWeight_EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR",
];

local leptonWeight_MU_List = [
  "leptonWeight_MUON_EFF_ISO_STAT",
  "leptonWeight_MUON_EFF_RECO_STAT",
  "leptonWeight_MUON_EFF_RECO_SYS",
  "leptonWeight_MUON_EFF_TTVA_STAT",
  "leptonWeight_MUON_EFF_TTVA_SYS",
];

# trigger weights
local trigweight_EL_list = [
  "trigWeight_EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR",
];

local trigWeight_MU_list = [
  "trigWeight_MUON_EFF_TrigStatUncertainty",
  "trigWeight_MUON_EFF_TrigSystUncertainty",
];


# V+jets theory

local Vjets2211_MUR_MUF_Scale = [
  "MUR0p5_MUF1_PDF303200_PSMUR0p5_PSMUF1",
  "MUR0p5_MUF0p5_PDF303200_PSMUR0p5_PSMUF0p5",
  "MUR1_MUF0p5_PDF303200_PSMUR1_PSMUF0p5",
  "MUR1_MUF2_PDF303200_PSMUR1_PSMUF2",
  "MUR2_MUF1_PDF303200_PSMUR2_PSMUF1",
  "MUR2_MUF2_PDF303200_PSMUR2_PSMUF2",
];

local Vjets_NNPDF30nnlo_hessian_PDF = [
  "MUR1_MUF1_PDF303%03d"%i for i in std.range(201,300)
];


# sinlgetop theory

local singletop_ren_fac_scale = [
  "muR0p50muF0p50",
  "muR0p50muF1p00",
  "muR0p50muF2p00",
  "muR1p00muF2p00",
  "muR1p00muF0p50",
  "muR2p00muF0p50",
  "muR2p00muF1p00",
  "muR2p00muF2p00",
];

# ttbar theory

local ttbar_matrix_element = ["aMcAtNloPy8_NoSys"];

local ttbar_parton_shower = ["PowhegHerwig713_NoSys"];

local ttbar_ren_fac_scale = [
  "muR0p5muF2p0",
  "muR0p5muF1p0",
  "muR0p5muF0p5",
  "muR1p0muF2p0",
  "muR1p0muF0p5",
  "muR2p0muF0p5",
  "muR2p0muF1p0",
  "muR2p0muF2p0",
];

local ttbar_ISR_scale = [
  "Var3cDown",
  "Var3cUp",
];

local ttbar_FSR_scale = [
  "isrmuRfac1p0_fsrmuRfac2p0",
  "isrmuRfac1p0_fsrmuRfac0p5",
];

local ttbar_NNPDF30_PDF = [
  "PDFset260%03d"%i for i in std.range(1,100)
];

# diboson theory

local Flat10Percent = ["1.10", "0.9"];

local diboson222_MUR_MUF_Scale = [
  "MUR0p5_MUF0p5_PDF261000",
  "MUR0p5_MUF1_PDF261000",
  "MUR1_MUF0p5_PDF261000",
  "MUR1_MUF2_PDF261000",
  "MUR2_MUF1_PDF261000",
  "MUR2_MUF2_PDF261000",
];

local diboson_NNPDF30_PDF = [
  "MUR1_MUF1_PDF261%03d"%i for i in std.range(1,101)
];



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
        "name" : name + "_NNPDF30nnlo_hessian",
        "wlist" : [LHE(x) for x in Vjets_NNPDF30nnlo_hessian_PDF],
        "handle" : "hessian",
        "sys_type" : "theory",
      } for name in ["wjets_2211", "zjets_2211"]
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
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["ttbar"]
    ]

    +
    [
      {
        "name" : name + "_Flat10Percent",
        "wlist" : Flat10Percent ,
        "handle" : "min_max",
        "sys_type" : "theory"
      } for name in ["diboson"]
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
    "diboson" : JET_EtaIntercalibration_NonClosure_List + trigweight_EL_list + trigWeight_MU_list + leptonWeight_EL_List + leptonWeight_MU_List + jvtWeight_List + MUON_List + EG_List + MET_SoftTrk_List + JET_JER_List + JET_NP_List + bTagWeight_List + add_process_prefix("diboson", ["MUR_MUF_Scale", "NNPDF30nnlo_hessian"]), # MUR_MUF has problem in mc16d diboson JET_NP_List + bTagWeight_List +  "MUR_MUF_Scale", "NNPDF30_PDF"
    "ttbar" : JET_EtaIntercalibration_NonClosure_List + trigweight_EL_list + trigWeight_MU_list + leptonWeight_EL_List + leptonWeight_MU_List + jvtWeight_List + MUON_List + EG_List + MET_SoftTrk_List + JET_JER_List + JET_NP_List + bTagWeight_List + ["ttbar_matrix_element", "ttbar_parton_shower"] + add_process_prefix("ttbar", ["ren_fac_scale","ISR_scale","FSR_scale","NNPDF30_PDF"]),
    "singletop" : JET_EtaIntercalibration_NonClosure_List + trigweight_EL_list + trigWeight_MU_list + leptonWeight_EL_List + leptonWeight_MU_List + jvtWeight_List + MUON_List + EG_List + MET_SoftTrk_List + JET_JER_List + JET_NP_List + bTagWeight_List + add_process_prefix("singletop", ["ren_fac_scale"]),
    #"wjets_2211" : JET_JER_List,
    #"zjets_2211" :JET_JER_List,
    #"diboson" : JET_JER_List,
    #"ttbar" :JET_JER_List,
    #"singletop" :JET_JER_List,
  }
}
