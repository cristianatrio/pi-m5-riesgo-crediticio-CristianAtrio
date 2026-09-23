# Reporte de data drift - escenario `sintetico`

Generado: 2026-09-23T11:17:16  |  referencia: 8,137 filas  |  actual: 2,000 filas

## Estado global: **DRIFT DETECTADO**

- Variables criticas (PSI >= 0.25): 3 ['huella_consulta', 'puntaje_datacredito', 'tendencia_ingresos']
- Variables en alerta (PSI >= 0.1): 2 ['salario_cliente', 'edad_cliente']
- Variables estables: 15
- Drift de prediccion: PSI 0.842 (CRITICO); tasa de riesgo 1.5% -> 1.7% al umbral 0.66
- Accion recomendada: Revisar origen de datos y evaluar reentrenamiento

- Tasa de mora: 4.78% -> 5.00% (+0.22 pp). Los creditos recientes pueden mostrar menos mora por censura (aun no vencieron), ver EDA 3.6.

## Detalle por variable

| Variable | Tipo | PSI | Nivel | Test | p-valor | Ref (mediana/moda) | Actual |
|---|---|---|---|---|---|---|---|
| huella_consulta | numerica | 2.025 | CRITICO | KS | 6.62e-321 | 4 | 7 |
| puntaje_datacredito | numerica | 0.768 | CRITICO | KS | 1.67e-171 | 791 | 744 |
| tendencia_ingresos | categorica | 0.715 | CRITICO | chi2 | 1.86e-303 | Creciente | Decreciente |
| salario_cliente | numerica | 0.239 | ALERTA | KS | 7.19e-51 | 3,000,000 | 2,365,612 |
| edad_cliente | numerica | 0.238 | ALERTA | KS | 4.69e-45 | 42 | 36 |
| promedio_ingresos_datacredito | numerica | 0.007 | OK | KS | 0.937 | 1,187,852 | 1,185,376 |
| capital_prestado | numerica | 0.006 | OK | KS | 0.788 | 1,902,000 | 1,897,945 |
| creditos_sectorFinanciero | numerica | 0.005 | OK | KS | 0.841 | 2 | 2 |
| saldo_principal | numerica | 0.004 | OK | KS | 0.808 | 14,310 | 14,221 |
| cant_creditosvigentes | numerica | 0.003 | OK | KS | 1 | 5 | 5 |
| total_otros_prestamos | numerica | 0.002 | OK | KS | 0.549 | 1,000,000 | 1,000,000 |
| plazo_meses | numerica | 0.002 | OK | KS | 0.972 | 10 | 10 |
| saldo_total | numerica | 0.002 | OK | KS | 0.967 | 16,193 | 16,123 |
| cuota_pactada | numerica | 0.002 | OK | KS | 0.89 | 183,354 | 181,900 |
| tipo_laboral | categorica | 0.001 | OK | chi2 | 0.197 | Empleado | Empleado |
| creditos_sectorReal | numerica | 0.001 | OK | KS | 1 | 1 | 1 |
| tipo_credito | categorica | 0.000 | OK | chi2 | 0.986 | 4 | 4 |
| creditos_sectorCooperativo | numerica | 0.000 | OK | KS | 1 | 0 | 0 |
| saldo_mora | numerica | 0.000 | OK | KS | 1 | 0 | 0 |
| saldo_mora_codeudor | numerica | 0.000 | OK | KS | 1 | 0 | 0 |

Figuras: `psi_sintetico.png`, `distribuciones_sintetico.png`
