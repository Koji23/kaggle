NAME=ml
VERSION=1.0.0
IMAGE=originalgremlin/$(NAME)

build:
	docker build -t $(IMAGE) -t $(IMAGE):$(VERSION) .

run:
	docker run -it --rm --hostname $(NAME) --name $(NAME) \
		-p 8888:8888 \
		-v $(CURDIR)/data:/usr/local/data \
		-v $(CURDIR)/notebooks:/usr/local/notebooks \
		$(IMAGE):$(VERSION)
