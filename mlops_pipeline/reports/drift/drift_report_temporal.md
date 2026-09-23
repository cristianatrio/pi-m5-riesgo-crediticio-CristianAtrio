# Reporte de data drift - escenario `temporal`

Generado: 2026-09-23T11:17:14  |  referencia: 7,951 filas  |  actual: 825 filas

## Estado global: **DRIFT DETECTADO**

- Variables criticas (PSI >= 0.25): 1 ['promedio_ingresos_datacredito']
- Variables en alerta (PSI >= 0.1): 3 ['total_otros_prestamos', 'salario_cliente', 'capital_prestado']
- Variables estables: 16
- Drift de prediccion: PSI 0.043 (OK); tasa de riesgo 1.5% -> 2.5% al umbral 0.66
- Accion recomendada: Revisar origen de datos y evaluar reentrenamiento

- Tasa de mora: 4.87% -> 3.52% (-1.35 pp). Los creditos recientes pueden mostrar menos mora por censura (aun no vencieron), ver EDA 3.6.

## Detalle por variable

| Variable | Tipo | PSI | Nivel | Test | p-valor | Ref (mediana/moda) | Actual |
|---|---|---|---|---|---|---|---|
| promedio_ingresos_datacredito | numerica | 0.449 | CRITICO | KS | 1.37e-34 | 1,188,034 | 1,424,803 |
| total_otros_prestamos | numerica | 0.232 | ALERTA | KS | 8.45e-19 | 1,000,000 | 739,000 |
| salario_cliente | numerica | 0.139 | ALERTA | KS | 1.93e-08 | 3,000,000 | 3,120,000 |
| capital_prestado | numerica | 0.136 | ALERTA | KS | 2.52e-18 | 1,896,000 | 2,494,800 |
| cuota_pactada | numerica | 0.070 | OK | KS | 1.68e-07 | 182,701 | 200,924 |
| tipo_laboral | categorica | 0.046 | OK | chi2 | 1.59e-08 | Empleado | Empleado |
| huella_consulta | numerica | 0.045 | OK | KS | 0.00808 | 4 | 3 |
| edad_cliente | numerica | 0.045 | OK | KS | 2.87e-05 | 42 | 39 |
| plazo_meses | numerica | 0.038 | OK | KS | 1.97e-19 | 10 | 10 |
| saldo_principal | numerica | 0.019 | OK | KS | 0.213 | 14,348 | 13,878 |
| puntaje_datacredito | numerica | 0.019 | OK | KS | 0.108 | 791 | 787 |
| saldo_total | numerica | 0.012 | OK | KS | 0.227 | 16,262 | 14,795 |
| cant_creditosvigentes | numerica | 0.009 | OK | KS | 0.915 | 5 | 5 |
| tendencia_ingresos | categorica | 0.009 | OK | chi2 | 0.0866 | Creciente | Creciente |
| creditos_sectorFinanciero | numerica | 0.008 | OK | KS | 0.425 | 2 | 2 |
| creditos_sectorCooperativo | numerica | 0.003 | OK | KS | 0.826 | 0 | 0 |
| tipo_credito | categorica | 0.001 | OK | chi2 | 0.958 | 4 | 4 |
| creditos_sectorReal | numerica | 0.000 | OK | KS | 1 | 1 | 1 |
| saldo_mora | numerica | 0.000 | OK | KS | 1 | 0 | 0 |
| saldo_mora_codeudor | numerica | 0.000 | OK | KS | 1 | 0 | 0 |

Figuras: `psi_temporal.png`, `distribuciones_temporal.png`
