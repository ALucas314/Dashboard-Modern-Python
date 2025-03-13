DELIMITER $$

CREATE TRIGGER historico_clima_trigger
AFTER INSERT ON dados_atuais -- A trigger é acionada após uma inserção na tabela `dados_atuais`
FOR EACH ROW
BEGIN
    -- Inserir os dados na tabela `historico_clima`
    INSERT INTO historico_clima (
        `data`, `dia_semana`, `previsao`, `temperatura_min`, `temperatura_max`, `probabilidade_chuva`, 
        `descricao_dia`, `descricao_madrugada`, `descricao_manha`, `descricao_tarde`, `descricao_noite`, 
        `chuva_mm`, `vento_direcao`, `vento_velocidade`, `umidade_min`, `umidade_max`, `arco_iris`, 
        `sol_nascer`, `sol_por`, `lua_fase`
    ) VALUES 
    (
        NEW.data, -- Usando os dados da nova linha inserida na tabela `dados_atuais`
        NEW.dia_semana,
        NEW.previsao,
        NEW.temperatura_min,
        NEW.temperatura_max,
        NEW.probabilidade_chuva,
        NEW.descricao_dia,
        NEW.descricao_madrugada,
        NEW.descricao_manha,
        NEW.descricao_tarde,
        NEW.descricao_noite,
        NEW.chuva_mm,
        NEW.vento_direcao,
        NEW.vento_velocidade,
        NEW.umidade_min,
        NEW.umidade_max,
        NEW.arco_iris,
        NEW.sol_nascer,
        NEW.sol_por,
        NEW.lua_fase
    );
END$$

DELIMITER ;