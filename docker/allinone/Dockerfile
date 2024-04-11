ARG BASE_IMAGE="eosphorosai/dbgpt-openai"

FROM ${BASE_IMAGE}

RUN pip3 install dashscope

RUN apt-get update && apt-get install -y wget gnupg lsb-release net-tools

COPY docker/allinone/allinone-entrypoint.sh /usr/local/bin/allinone-entrypoint.sh
# COPY docker/examples/sqls/*_mysql.sql /docker-entrypoint-initdb.d/

RUN mkdir /sql_log

ENTRYPOINT ["/usr/local/bin/allinone-entrypoint.sh"]