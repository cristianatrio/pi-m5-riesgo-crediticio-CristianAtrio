# Reporte de data drift - escenario `temporal`

Generado: 2026-09-16T09:21:29  |  referencia: 7,948 filas  |  actual: 825 filas

## Estado global: **DRIFT DETECTADO**

- Variables criticas (PSI >= 0.25): 1 ['promedio_ingresos_datacredito']
- Variables en alerta (PSI >= 0.1): 3 ['total_otros_prestamos', 'capital_prestado', 'salario_cliente']
- Variables estables: 16
- Drift de prediccion: PSI 0.037 (OK); tasa de riesgo 9.2% -> 11.2% al umbral 0.50
- Accion recomendada: Revisar origen de datos y evaluar reentrenamiento

- Tasa de mora: 4.88% -> 3.52% (-1.37 pp). Los creditos recientes pueden mostrar menos mora por censura (aun no vencieron), ver EDA 3.6.

## Detalle por variable

| Variable | Tipo | PSI | Nivel | Test | p-valor | Ref (mediana/moda) | Actual |
|---|---|---|---|---|---|---|---|
| promedio_ingresos_datacredito | numerica | 0.583 | CRITICO | KS | 1.14e-35 | 1,165,583 | 1,424,803 |
| total_otros_prestamos | numerica | 0.223 | ALERTA | KS | 2.01e-19 | 1,000,000 | 739,000 |
| capital_prestado | numerica | 0.146 | ALERTA | KS | 5.2e-19 | 1,869,251 | 2,494,800 |
| salario_cliente | numerica | 0.143 | ALERTA | KS | 7.43e-09 | 3,000,000 | 3,120,000 |
| cuota_pactada | numerica | 0.058 | OK | KS | 1.7e-07 | 180,814 | 200,924 |
| tipo_laboral | categorica | 0.048 | OK | chi2 | 6.59e-09 | Empleado | Empleado |
| edad_cliente | numerica | 0.045 | OK | KS | 3.06e-05 | 42 | 39 |
| huella_consulta | numerica | 0.041 | OK | KS | 0.00711 | 4 | 3 |
| plazo_meses | numerica | 0.034 | OK | KS | 4.18e-20 | 10 | 10 |
| saldo_principal | numerica | 0.022 | OK | KS | 0.206 | 14,412 | 13,878 |
| puntaje_datacredito | numerica | 0.020 | OK | KS | 0.117 | 791 | 787 |
| saldo_total | numerica | 0.016 | OK | KS | 0.207 | 16,193 | 14,795 |
| tendencia_ingresos | categorica | 0.008 | OK | chi2 | 0.0984 | Creciente | Creciente |
| cant_creditosvigentes | numerica | 0.008 | OK | KS | 0.961 | 5 | 5 |
| creditos_sectorFinanciero | numerica | 0.006 | OK | KS | 0.557 | 2 | 2 |
| creditos_sectorCooperativo | numerica | 0.003 | OK | KS | 0.919 | 0 | 0 |
| tipo_credito | categorica | 0.001 | OK | chi2 | 0.96 | 4 | 4 |
| creditos_sectorReal | numerica | 0.001 | OK | KS | 1 | 1 | 1 |
| saldo_mora | numerica | 0.000 | OK | KS | 1 | 0 | 0 |
| saldo_mora_codeudor | numerica | 0.000 | OK | KS | 1 | 0 | 0 |

Figuras: `psi_temporal.png`, `distribuciones_temporal.png`
