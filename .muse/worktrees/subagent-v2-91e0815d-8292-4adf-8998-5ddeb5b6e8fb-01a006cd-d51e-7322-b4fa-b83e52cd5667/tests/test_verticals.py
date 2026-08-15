def test_verticals():
    from matgraph.verticals.battery import battery_metrics
    from matgraph.verticals.catalysis import catalysis_metrics
    from matgraph.verticals.photovoltaics import pv_metrics
    from matgraph.verticals.thermoelectrics import thermo_metrics
    from matgraph.verticals.twod import twod_metrics
    from matgraph.verticals.alloys import alloy_metrics
    from matgraph.verticals.defects import defect_metrics
    assert battery_metrics("LiFePO4", formation_energy_per_atom=-1.2)["theoretical_capacity_mah_g"]>100
    assert "d_band_center_eV_proxy" in catalysis_metrics("Pt")
    assert pv_metrics("Si", band_gap=1.1)["sq_limit_percent_proxy"] is not None
    assert thermo_metrics("Bi2Te3", band_gap=0.2)["zt_proxy"] is not None
    assert alloy_metrics("CoCrFeMnNi")["n_elements"]==5
    assert defect_metrics("Si", formation_energy_per_atom=-0.5)["vacancy_formation_eV_proxy"] is not None
