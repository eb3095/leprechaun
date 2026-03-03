{{/*
Expand the name of the chart.
*/}}
{{- define "leprechaun.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "leprechaun.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "leprechaun.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "leprechaun.labels" -}}
helm.sh/chart: {{ include "leprechaun.chart" . }}
{{ include "leprechaun.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: leprechaun
{{- end }}

{{/*
Selector labels
*/}}
{{- define "leprechaun.selectorLabels" -}}
app.kubernetes.io/name: {{ include "leprechaun.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API component labels
*/}}
{{- define "leprechaun.apiLabels" -}}
{{ include "leprechaun.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "leprechaun.apiSelectorLabels" -}}
{{ include "leprechaun.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Redis component labels
*/}}
{{- define "leprechaun.redisLabels" -}}
{{ include "leprechaun.labels" . }}
app.kubernetes.io/component: cache
{{- end }}

{{/*
Redis selector labels
*/}}
{{- define "leprechaun.redisSelectorLabels" -}}
{{ include "leprechaun.selectorLabels" . }}
app.kubernetes.io/component: cache
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "leprechaun.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "leprechaun.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the namespace name
*/}}
{{- define "leprechaun.namespace" -}}
{{- default .Release.Namespace .Values.namespace.name }}
{{- end }}

{{/*
Redis URL
*/}}
{{- define "leprechaun.redisUrl" -}}
{{- if .Values.redis.enabled -}}
redis://{{ include "leprechaun.fullname" . }}-redis.{{ include "leprechaun.namespace" . }}.svc.cluster.local:6379/0
{{- else -}}
{{- .Values.config.redisUrl -}}
{{- end -}}
{{- end }}

{{/*
ConfigMap name
*/}}
{{- define "leprechaun.configMapName" -}}
{{ include "leprechaun.fullname" . }}-config
{{- end }}

{{/*
Secret name
*/}}
{{- define "leprechaun.secretName" -}}
{{ include "leprechaun.fullname" . }}-secrets
{{- end }}

{{/*
Redis name
*/}}
{{- define "leprechaun.redisName" -}}
{{ include "leprechaun.fullname" . }}-redis
{{- end }}
