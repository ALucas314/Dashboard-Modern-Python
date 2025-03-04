CREATE OR REPLACE VIEW view_clima_completo AS
SELECT 
    dh.Data, 
    dh.`Hora UTC`, 
    t.`TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)` AS Temperatura,
    t.`TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)` AS Temp_Max,
    t.`TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)` AS Temp_Min,
    t.`TEMPERATURA DO PONTO DE ORVALHO (°C)` AS Temp_Orvalho,
    t.`TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)` AS Temp_Orvalho_Max,
    t.`TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)` AS Temp_Orvalho_Min,
    p.`PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)` AS Pressao,
    u.`UMIDADE RELATIVA DO AR, HORARIA (%)` AS Umidade,
    v.`VENTO, VELOCIDADE HORARIA (m/s)` AS Vento_Velocidade,
    v.`VENTO, DIREÇÃO HORARIA (gr) (° (gr))` AS Vento_Direcao,
    o.`RADIACAO GLOBAL (Kj/m²)` AS Radiacao,
    o.`PRECIPITAÇÃO TOTAL, HORÁRIO (mm)` AS Precipitacao
FROM 
    data_hora dh
LEFT JOIN 
    Temperaturas t ON dh.id_data_hora = t.data_hora_id_data_hora
LEFT JOIN 
    Pressao p ON dh.id_data_hora = p.data_hora_id_data_hora
LEFT JOIN 
    Umidade u ON dh.id_data_hora = u.data_hora_id_data_hora
LEFT JOIN 
    ventos v ON dh.id_data_hora = v.data_hora_id_data_hora
LEFT JOIN 
    Outras_informacoes o ON dh.id_data_hora = o.data_hora_id_data_hora;
