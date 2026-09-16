# Reporte de data drift - escenario `sintetico`

Generado: 2026-09-16T09:21:32  |  referencia: 8,610 filas  |  actual: 2,000 filas

## Estado global: **DRIFT DETECTADO**

- Variables criticas (PSI >= 0.25): 3 ['huella_consulta', 'puntaje_datacredito', 'tendencia_ingresos']
- Variables en alerta (PSI >= 0.1): 2 ['salario_cliente', 'edad_cliente']
- Variables estables: 15
- Drift de prediccion: PSI 0.683 (CRITICO); tasa de riesgo 9.3% -> 23.8% al umbral 0.50
- Accion recomendada: Revisar origen de datos y evaluar reentrenamiento

- Tasa de mora: 4.75% -> 5.00% (+0.25 pp). Los creditos recientes pueden mostrar menos mora por censura (aun no vencieron), ver EDA 3.6.

## Detalle por variable

| Variable | Tipo | PSI | Nivel | Test | p-valor | Ref (mediana/moda) | Actual |
|---|---|---|---|---|---|---|---|
| huella_consulta | numerica | 2.063 | CRITICO | KS | 6.75e-321 | 4 | 7 |
| puntaje_datacredito | numerica | 0.749 | CRITICO | KS | 4.25e-171 | 791 | 744 |
| tendencia_ingresos | categorica | 0.696 | CRITICO | chi2 | 3.21e-301 | Creciente | Decreciente |
| salario_cliente | numerica | 0.241 | ALERTA | KS | 1.75e-51 | 3,000,000 | 2,365,612 |
| edad_cliente | numerica | 0.230 | ALERTA | KS | 5.94e-43 | 42 | 36 |
| capital_prestado | numerica | 0.006 | OK | KS | 0.765 | 1,908,000 | 1,897,945 |
| promedio_ingresos_datacredito | numerica | 0.005 | OK | KS | 0.936 | 1,187,852 | 1,185,376 |
| saldo_principal | numerica | 0.004 | OK | KS | 0.71 | 14,411 | 14,221 |
| cuota_pactada | numerica | 0.004 | OK | KS | 0.74 | 182,638 | 181,900 |
| creditos_sectorFinanciero | numerica | 0.004 | OK | KS | 0.989 | 2 | 2 |
| plazo_meses | numerica | 0.003 | OK | KS | 0.975 | 10 | 10 |
| cant_creditosvigentes | numerica | 0.003 | OK | KS | 0.995 | 5 | 5 |
| total_otros_prestamos | numerica | 0.002 | OK | KS | 0.65 | 1,000,000 | 1,000,000 |
| saldo_total | numerica | 0.002 | OK | KS | 0.929 | 16,143 | 16,123 |
| creditos_sectorReal | numerica | 0.001 | OK | KS | 0.996 | 1 | 1 |
| tipo_laboral | categorica | 0.001 | OK | chi2 | 0.355 | Empleado | Empleado |
| tipo_credito | categorica | 0.000 | OK | chi2 | 0.955 | 4 | 4 |
| creditos_sectorCooperativo | numerica | 0.000 | OK | KS | 1 | 0 | 0 |
| saldo_mora | numerica | 0.000 | OK | KS | 1 | 0 | 0 |
| saldo_mora_codeudor | numerica | 0.000 | OK | KS | 1 | 0 | 0 |

Figuras: `psi_sintetico.png`, `distribuciones_sintetico.png`
