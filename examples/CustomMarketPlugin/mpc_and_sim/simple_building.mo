within ;
model mpc_room_example
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor roomCapacity(C=10000, T(start=
          294.15))
    annotation (Placement(transformation(extent={{-30,10},{-10,30}})));
  Modelica.Thermal.HeatTransfer.Components.Convection convection
    annotation (Placement(transformation(extent={{2,-10},{22,10}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature
                                                         ambientTemperature
                annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={50,0})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow
    annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
  Modelica.Blocks.Sources.Constant convection_coefficient(k=5)
    annotation (Placement(transformation(extent={{40,20},{20,40}})));
  Modelica.Blocks.Interfaces.RealInput Q_in
    annotation (Placement(transformation(extent={{-120,-20},{-80,20}})));
  Modelica.Blocks.Sources.Sine ambient_Temperature(
    amplitude=5,
    f=1/86400,
    offset=278.15)
    annotation (Placement(transformation(extent={{100,-10},{80,10}})));
equation
  connect(roomCapacity.port, convection.solid) annotation (Line(points={{-20,10},
          {-20,0},{2,0}},                     color={191,0,0}));
  connect(ambientTemperature.port, convection.fluid)
    annotation (Line(points={{40,1.33227e-15},{34,1.33227e-15},{34,0},{22,0}},
                                               color={191,0,0}));
  connect(prescribedHeatFlow.port, roomCapacity.port)
    annotation (Line(points={{-40,0},{-20,0},{-20,10}},    color={191,0,0}));
  connect(convection_coefficient.y, convection.Gc)
    annotation (Line(points={{19,30},{12,30},{12,10}},color={0,0,127}));
  connect(ambientTemperature.T, ambient_Temperature.y)
    annotation (Line(points={{62,-1.33227e-15},{72,-1.33227e-15},{72,0},{79,0}},
                                             color={0,0,127}));
  connect(Q_in, prescribedHeatFlow.Q_flow) annotation (Line(points={{-100,0},{
          -60,0}},                   color={0,0,127}));
  annotation (uses(Modelica(version="4.0.0")),
    Diagram(coordinateSystem(extent={{-120,-60},{120,60}})),
    Icon(coordinateSystem(extent={{-120,-60},{120,60}})),
    version="1",
    conversion(from(version="", script=
            "modelica://mpc_room_example/ConvertFrommpc_room_example_.mos")));
end mpc_room_example;
