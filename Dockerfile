#Download base image ubuntu 16.04
FROM ubuntu:16.04

FROM python:3

# install wkhtmltoimage
RUN cd ~
RUN wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.3/wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
RUN tar vxf wkhtmltox-0.12.3_linux-generic-amd64.tar.xz 
RUN cp wkhtmltox/bin/wk* /usr/local/bin/

# install vi
RUN apt-get -y update
RUN apt-get install -y vim

# RUN apt-get install -y xvfb

WORKDIR /MagickTable
ADD . /MagickTable
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN python manage.py makemigrations
RUN python manage.py migrate

# Remove this!
RUN chmod -R 777 /MagickTable/media


# Make port 8000 available to the world outside this container
EXPOSE 8000

# CMD python manage.py runserver
CMD python manage.py runserver 0.0.0.0:8000