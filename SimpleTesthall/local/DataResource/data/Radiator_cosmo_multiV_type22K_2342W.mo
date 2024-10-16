within Test_inter_teaser.Test_EFH_urbanrenet.Test_EFH_urbanrenet_DataBase;
record Radiator_cosmo_multiV_type22K_2342W
  "-"
  extends AixLib.DataBase.Radiators.RadiatorBaseDataDefinition(
      NominalPower=780.667,
      RT_nom={328.15,318.15,293.15},
      PressureDrop=1017878,
      Exponent=1.334,
      VolumeWater=6.1,
      MassSteel=21.17,
      DensitySteel=7900,
      CapacitySteel=551,
      LambdaSteel=60,
      Type=AixLib.Fluid.HeatExchangers.Radiators.BaseClasses.RadiatorTypes.PanelRadiator22,
      length=3,
      height=0.5)

  annotation ();
end Radiator_cosmo_multiV_type22K_2342W;
