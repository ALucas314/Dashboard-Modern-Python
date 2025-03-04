DELIMITER //

CREATE PROCEDURE CalcularUmidadeMediaPorDia()
BEGIN
    SELECT dh.Data, AVG(u.`UMIDADE RELATIVA DO AR, HORARIA (%)`) AS umidade_media
    FROM Umidade u
    JOIN data_hora dh ON u.data_hora_id_data_hora = dh.id_data_hora
    GROUP BY dh.Data;
END //

DELIMITER ;