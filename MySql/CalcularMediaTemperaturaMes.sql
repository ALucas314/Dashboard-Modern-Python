DELIMITER //

CREATE FUNCTION CalcularMediaTemperaturaMes(p_mes INT, p_ano INT) 
RETURNS DECIMAL(5,2)
DETERMINISTIC
BEGIN
    DECLARE media_temp DECIMAL(5,2);

    -- Calcular a média da temperatura para o mês e ano especificados
    SELECT AVG(`TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)`)
    INTO media_temp
    FROM Temperaturas t
    JOIN data_hora dh ON t.data_hora_id_data_hora = dh.id_data_hora
    WHERE MONTH(dh.Data) = p_mes AND YEAR(dh.Data) = p_ano;

    -- Retornar a média calculada
    RETURN media_temp;
END //

DELIMITER ;

SELECT CalcularMediaTemperaturaMes(1, 2025) AS MediaTemperatura;