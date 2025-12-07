def test_chains(fake_chains, fakeR_chains):
    assert fake_chains.shape == fakeR_chains.shape

def test_X(fake_X, fakeR_X):
    assert fake_X.shape == fakeR_X.shape

def test_Z(fake_Z, fakeR_Z):
    assert fake_Z.shape == fakeR_Z.shape

def test_ARs(fake_ARs, fakeR_ARs):
    assert fake_ARs.shape == fakeR_ARs.shape

def test_Rhats(fake_Rhats, fakeR_Rhats):
    assert fake_Rhats.shape == fakeR_Rhats.shape

def test_chainburned(fake_chainburned, fakeR_chainburned):
    assert fake_chainburned.shape == fakeR_chainburned.shape

def test_zchainsummary(fake_zchainsummary, fakeR_zchainsummary):
    assert fake_zchainsummary.shape == fakeR_zchainsummary.shape

def test_Rs(fake_Rs, fakeR_Rs):
    assert fake_Rs.shape == fakeR_Rs.shape

def test_probloc(fake_probloc, fakeR_probloc):
    assert fake_probloc.shape == fakeR_probloc.shape

def test_optsed_costs(fake_optsed_costs, fakeR_optsed_costs):
    assert fake_optsed_costs.shape == fakeR_optsed_costs.shape

def test_optsed_Qs(fake_optsed_Qs, fakeR_optsed_Qs):
    assert fake_optsed_Qs.shape == fakeR_optsed_Qs.shape

def test_optsed_accQ(fake_optsed_accQ, fakeR_optsed_accQ):
    assert fake_optsed_accQ.shape == fakeR_optsed_accQ.shape

def test_accFOR(fake_accFOR, fakeR_accFOR):
    assert fake_accFOR.shape == fakeR_accFOR.shape

def test_maxPSR(fake_maxPSR, fakeR_maxPSR):
    assert fake_maxPSR.shape == fakeR_maxPSR.shape