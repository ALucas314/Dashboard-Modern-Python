ALTER TABLE dados_atuais
ADD FULLTEXT INDEX idx_fulltext_descricao (descricao_dia, descricao_madrugada, descricao_manha, descricao_tarde, descricao_noite, previsao);

ALTER TABLE historico_clima
ADD FULLTEXT INDEX idx_fulltext_descricao (descricao_dia, descricao_madrugada, descricao_manha, descricao_tarde, descricao_noite, previsao);



SELECT *
FROM dados_atuais
WHERE MATCH(descricao_dia, descricao_madrugada, descricao_manha, descricao_tarde, descricao_noite, previsao)
AGAINST('chuvoso' IN NATURAL LANGUAGE MODE);

SELECT *
FROM historico_clima
WHERE MATCH(descricao_dia, descricao_madrugada, descricao_manha, descricao_tarde, descricao_noite, previsao)
AGAINST('temporal' IN NATURAL LANGUAGE MODE);

SELECT *
FROM dados_atuais
WHERE MATCH(descricao_dia, descricao_madrugada, descricao_manha, descricao_tarde, descricao_noite, previsao)
AGAINST('+chuvoso -temporal' IN BOOLEAN MODE);