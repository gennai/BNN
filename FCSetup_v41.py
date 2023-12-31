gennu = '''
TLorentzVector nus;
nus.SetPtEtaPhiE(genNuTot_pt, genNuTot_eta, genNuTot_phi, genNuTot_e);
return nus.{}();
'''
gennu1 = '''
TLorentzVector nus;
nus.SetPtEtaPhiE(genNu1_pt, genNu1_eta, genNu1_phi, genNu1_e);
return nus.{}();
'''
gennu2 = '''
TLorentzVector nus;
nus.SetPtEtaPhiE(genNu2_pt, genNu2_eta, genNu2_phi, genNu2_e);
return nus.{}();
'''

tau1 = '''
TLorentzVector tau1;
tau1.SetPtEtaPhiE(dau1_pt, dau1_eta, dau1_phi, dau1_e);
return tau1.{}();
'''
tau2 = '''
TLorentzVector tau1;
tau1.SetPtEtaPhiE(dau2_pt, dau2_eta, dau2_phi, dau2_e);
return tau1.{}();
'''
BRANCHES = {
  # ROOT features
  'eventNumber'                   : ("EventNumber"                                        , 'int32' ),
  'tau1_px'                   : (tau1.format('Px')                                        , 'float32' ),
  'tau1_py'                  : (tau1.format('Py')                                        , 'float32' ),
  'tau1_pz'                   : (tau1.format('Pz')                                        , 'float32' ),
  'tau1_e'                     : (tau1.format('E')                                        , 'float32' ),
  'tau2_px'                   : (tau2.format('Px')                                        , 'float32' ),
  'tau2_py'                  : (tau2.format('Py')                                        , 'float32' ),
  'tau2_pz'                   : (tau2.format('Pz')                                        , 'float32' ),
  'tau2_e'                     : (tau2.format('E')                                        , 'float32' ),
  'tau1_pt'                   : ('dau1_pt'                                                , 'float32' ),
  'tau1_eta'                  : ('dau1_eta'                                               , 'float32' ),
  'tau1_phi'                  : ('dau1_phi'                                               , 'float32' ),
  'tau2_pt'                   : ('dau2_pt'                                                , 'float32' ),
  'tau2_eta'                  : ('dau2_eta'                                               , 'float32' ),
  'tau2_phi'                  : ('dau2_phi'                                               , 'float32' ),
  'tau1_dm'                  : ('dau1_decayMode'                                                 , 'float32' ),
  'tau2_dm'                  : ('dau2_decayMode'                                                 , 'float32' ),
  'ditau_deltaphi'           : ('ditau_deltaPhi'                                          , 'float32' ), 
  'ditau_deltaeta'           : ('ditau_deltaEta'                                          , 'float32' ), 
  #'MHT_x'                    : ('MHTx'                                                    , 'float32' ),
  #'MHT_y'                    : ('MHTy'                                                    , 'float32' ),
  'MET_pt'                    : ('met_et'                                                 , 'float32' ),
  'MET_phi'                   : ('met_phi>=M_PI?met_phi-2.0*M_PI:met_phi'                 , 'float32' ),
  'MET_X'                    : ('METx'                                                    , 'float32' ),
  'MET_Y'                    : ('METy'                                                    , 'float32' ),
  #'METnoM_x'                  : ('METnoMux'                                               , 'float32' ),
  #'METnoM_y'                    : ('METnoMuy'                                             , 'float32' ),
  'DeepMET_ResponseTune_px' : ('DeepMET_ResponseTune_px'                                   , 'float32' ), 
  'DeepMET_ResponseTune_py' : ('DeepMET_ResponseTune_py'                                   , 'float32' ), 
  'DeepMET_ResolutionTune_px' : ('DeepMET_ResolutionTune_px'                                   , 'float32' ), 
  'DeepMET_ResolutionTune_py' : ('DeepMET_ResolutionTune_py'                                   , 'float32' ), 
  'MET_covXX'                 : ('met_cov00'                                              , 'float32' ),
  'MET_covXY'                 : ('met_cov01'                                              , 'float32' ),
  'MET_covYY'                 : ('met_cov11'                                              , 'float32' ),
  'PU_npvs'                   : ('npv'                                                    , 'int32'   ),
  'bjet1_pt'                  : ('bjet1_pt'                                               , 'float32' ),
  'bjet1_eta'                 : ('bjet1_eta'                                              , 'float32' ),
  'bjet1_phi'                 : ('bjet1_phi'                                              , 'float32' ),
  'bjet1_deepFlavor'          : ('bjet1_bID_deepFlavor'                                   , 'float32' ),
  'bjet1_pNet_b'          : ('bjet1_pnet_bb + bjet1_pnet_b'                                   , 'float32' ),
  'bjet1_pNet_c'          : ('bjet1_pnet_cc + bjet1_pnet_c'                                   , 'float32' ),
  'bjet1_pNet_uds'          : ('bjet1_pnet_uds + bjet1_pnet_g'                                   , 'float32' ),
  'bjet2_pt'                  : ('bjet2_pt'                                               , 'float32' ),
  'bjet2_eta'                 : ('bjet2_eta'                                              , 'float32' ),
  'bjet2_phi'                 : ('bjet2_phi'                                              , 'float32' ),
  'bjet2_deepFlavor'          : ('bjet2_bID_deepFlavor'                                   , 'float32' ),
  'bjet2_pNet_b'          : ('bjet2_pnet_bb + bjet2_pnet_b'                                   , 'float32' ),
  'bjet2_pNet_c'          : ('bjet2_pnet_cc + bjet2_pnet_c'                                   , 'float32' ),
  'bjet2_pNet_uds'          : ('bjet2_pnet_uds + bjet2_pnet_g'                                , 'float32' ),
   #'VBFjet1_pt'                : ('VBFjet1_pt>0?VBFjet1_pt:0'                              , 'float32' ),
  #'VBFjet1_eta'               : ('VBFjet1_pt>0?VBFjet1_eta:0'                             , 'float32' ),
  #'VBFjet1_phi'               : ('VBFjet1_pt>0?VBFjet1_phi:0'                             , 'float32' ),
  #'VBFjet1_deepFlavor'        : ('VBFjet1_pt>0?VBFjet1_btag_deepFlavor:0'                 , 'float32' ),
  #'VBFjet2_pt'                : ('VBFjet1_pt>0?VBFjet2_pt:0'                              , 'float32' ),
  #'VBFjet2_eta'               : ('VBFjet1_pt>0?VBFjet2_eta:0'                             , 'float32' ),
  #'VBFjet2_phi'               : ('VBFjet1_pt>0?VBFjet2_phi:0'                             , 'float32' ),
  #'VBFjet2_deepFlavor'        : ('VBFjet1_pt>0?VBFjet2_btag_deepFlavor:0'                 , 'float32' ),
  'tauH_SVFIT_mass'           : ('tauH_SVFIT_mass'                                        , 'float32' ),
  'pairType'                  : ('pairType'                                               , 'int32'   ),
  'is_test'                   : ('false'                                                  , 'bool'    ),
  'is_train'                  : ('false'                                                  , 'bool'    ),
  'is_valid'                  : ('false'                                                  , 'bool'    ),
  'sample_weight'             : ('1.0'                                                      , 'float32' ),
  'sample_class1'              : ('0'                                                      , 'int32' ),
  'sample_class2'              : ('0'                                                      , 'int32' ),
  'sample_class3'              : ('0'                                                      , 'int32' ),
  'sample_class4'              : ('0'                                                      , 'int32' ),
  ## dependent expressions
  'mT1'                       : ('sqrt(2*tau1_pt*MET_pt *(1-cos(tau1_phi-MET_phi )))'     , 'float32' ),
  'mT2'                       : ('sqrt(2*tau2_pt*MET_pt *(1-cos(tau2_phi-MET_phi )))'     , 'float32' ),
  'mTtt'                      : ('sqrt(2*tau1_pt*tau2_pt*(1-cos(tau1_phi-tau2_phi)))'     , 'float32' ),
  'mTtot'                     : ('sqrt(mT1*mT1+mT2*mT2+mTtt*mTtt)'                        , 'float32' ),
  'mVis'                      : ('sqrt(pow((tau1_e+tau2_e),2)-pow((tau1_px+tau2_px),2)-pow((tau1_py+tau2_py),2)-pow((tau1_pz+tau2_pz),2))' , 'float32' ),
  ## gen. neutrinos info
  'genNu_px'                  : (gennu.format('Px'), 'float32'),
  'genNu_py'                  : (gennu.format('Py'), 'float32'),
  'genNu_pz'                  : (gennu.format('Pz'), 'float32'),
  'genNu_e'                   : (gennu.format('E' ), 'float32'),
  'genNu1_px'                 : (gennu1.format('Px'), 'float32'),
  'genNu1_py'                 : (gennu1.format('Py'), 'float32'),
  'genNu1_pz'                 : (gennu1.format('Pz'), 'float32'),
  'genNu1_e'                  : (gennu1.format('E' ), 'float32'),
  'genNu2_px'                 : (gennu2.format('Px'), 'float32'),
  'genNu2_py'                 : (gennu2.format('Py'), 'float32'),
  'genNu2_pz'                 : (gennu2.format('Pz'), 'float32'),
  'genNu2_e'                  : (gennu2.format('E' ), 'float32'),
  #'genNu_m'                   : (gennu.format('M' ), 'float32'),
  'mGenReco'                  : ('sqrt(pow((tau1_e+tau2_e+genNu_e),2)-pow((tau1_px+tau2_px+genNu_px),2)-pow((tau1_py+tau2_py+genNu_py),2)-pow((tau1_pz+tau2_pz+genNu_pz),2))' , 'float32' ),  
  'HHKin_mass'                : ("HHKin_mass"                                               , 'float32' ),
  'k_mass'                    : ("mGenReco/mVis"                                               , 'float32' ),
  # lambda features
  #'sample_class'              : ('[0.,0.,0.,0.]'                                                      , 'float32' ),
  'channel'                   : (lambda x: {0:'mt', 1:'et', 2:'tt'}[x['pairType']]                       , None   ),
  'N_neutrinos'               : (lambda x: {'tt':2, 'mt':3, 'et':3, 'mm':4, 'em':4, 'ee':4}[x['channel']], 'int16'),
  "BDT_channel":          ("BDT_channel"      ,"float32"),
  "BDT_ditau_deltaPhi":          ("BDT_ditau_deltaPhi"      ,"float32"),
  "BDT_dib_abs_deltaPhi":          ("BDT_dib_abs_deltaPhi"      ,"float32"),
  "BDT_dib_deltaPhi":          ("BDT_dib_deltaPhi"      ,"float32"),
  "BDT_dau1MET_deltaPhi":          ("BDT_dau1MET_deltaPhi"      ,"float32"),
#"BDT_tauHsvfitMet_abs_deltaPhi":          ("BDT_tauHsvfitMet_abs_deltaPhi"      ,"float32"),
#"BDT_tauHsvfitMet_deltaPhi":          ("BDT_tauHsvfitMet_deltaPhi"      ,"float32"),
"BDT_bHMet_deltaPhi":          ("BDT_bHMet_deltaPhi"      ,"float32"),
#"BDT_HHsvfit_abs_deltaPhi":          ("BDT_HHsvfit_abs_deltaPhi"      ,"float32"),
"BDT_HT20":          ("BDT_HT20"      ,"float32"),
"BDT_topPairMasses":          ("BDT_topPairMasses"      ,"float32"),
"BDT_topPairMasses2":          ("BDT_topPairMasses2"      ,"float32"),
#"BDT_MX":          ("BDT_MX"      ,"float32"),
"BDT_bH_tauH_MET_InvMass":          ("BDT_bH_tauH_MET_InvMass"      ,"float32"),
#"BDT_bH_tauH_SVFIT_InvMass":          ("BDT_bH_tauH_SVFIT_InvMass"      ,"float32"),
"BDT_bH_tauH_InvMass":          ("BDT_bH_tauH_InvMass"      ,"float32"),
"BDT_total_CalcPhi":          ("BDT_total_CalcPhi"      ,"float32"),
"BDT_ditau_CalcPhi":          ("BDT_ditau_CalcPhi"      ,"float32"),
"BDT_dib_CalcPhi":          ("BDT_dib_CalcPhi"      ,"float32"),
#"BDT_MET_tauH_SVFIT_cosTheta":          ("BDT_MET_tauH_SVFIT_cosTheta"      ,"float32"),
"BDT_MET_bH_cosTheta":          ("BDT_MET_bH_cosTheta"      ,"float32"),
"BDT_b1_bH_cosTheta":          ("BDT_b1_bH_cosTheta"      ,"float32"),
#"BDT_tauH_SVFIT_reson_cosTheta":          ("BDT_tauH_SVFIT_reson_cosTheta"      ,"float32"),
}

FEATURES = [
  b for b in BRANCHES.keys() if not b in [
    'eventNumber',
    'pairType', 'is_train', 'is_valid', 'is_test', 'channel', 'sample_weight',
    'tauH_SVFIT_mass', 'target', 'genNu1_px', 'genNu1_py', 'genNu1_pz', 'genNu1_e',
    'genNu2_px', 'genNu2_py', 'genNu2_pz', 'genNu2_e',
    'genNu_px', 'genNu_py', 'genNu_pz', 'genNu_e',
    'mGenReco','sample_class1','sample_class2','sample_class3','sample_class4',
    'HHKin_mass',    
    #'tau1_phi','tau1_eta','tau1_pt','tau2_pt','tau2_phi','tau2_eta',
    #"tau1_dm","tau2_dm",
    "k_mass",
    #'tau1_px','tau1_py','tau1_pz','tau1_e','tau2_px','tau2_py','tau2_pz',"tau2_e",    
    #'MET_X',
    #"MET_Y",
    #"MET_covXX","MET_covYY","MET_covXY",
    #'mT1',"mT2",'mTtot'
  ]
]
#FEATURES = [b for b in BRANCHES.keys()]
SETUP = {
  'max_events': 50000,
  #'target'    : 'tauH_SVFIT_mass',
  #'target'    : ['genNu_px', 'genNu_py', 'genNu_pz', 'genNu_e'],
  'target'    : ['sample_class1','sample_class2'],
  #'target'    : ['sample_class1','sample_class2','sample_class3'],
  #'target'    : ['k_mass','sample_class1','sample_class3','sample_class4'],
}
