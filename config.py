config = dict(
    curtail_penalty=20, #($/MWh)
    ren_export_price=5, #($/MWh) 5 default
    min_locations=10,
    max_locations=10, 
    ren_penetration=0.9, 
    discount_rate=0.012,
    project_lifetime = 20,
    water_risk_penalty = 10**7,
    load_multiplier = 1.0
    #include_transmission_cost = True,
    #include_telecom_cost = True,
    #include_water_cost = True
)