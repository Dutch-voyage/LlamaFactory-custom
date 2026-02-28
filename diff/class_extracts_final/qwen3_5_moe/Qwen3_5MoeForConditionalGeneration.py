class Qwen3_5MoeForConditionalGeneration(Qwen3_5MoePreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3_5MoeConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5MoeModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        return self.model.get_video_features(
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs
        )

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        return self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3_5MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

        >>> model = Qwen3_5MoeForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-35B-A3B-Instruct", dtype="auto", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-35B-A3B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image in short."},
                ],
            }
        ]

        >>> # Preparation for inference
        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        >>> inputs = inputs.to(model.device)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=128)
        >>> generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        >>> processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "A woman in a plaid shirt sits on a sandy beach at sunset, smiling as she gives a high-five to a yellow Labrador Retriever wearing a harness. The ocean waves roll in the background."
        ```"""

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        return Qwen3_5MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # Overwritten -- requires 3D position ids

        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        # Early exit in case we are continuing generation from past kv
        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            text_positions += self.model.rope_deltas
            return text_positions

        # Otherwise compute 3d position ids for vision tokens and concat with text position ids
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if is_input_ids and (
            model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None
        ):
            model_kwargs = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
            vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)
            self.model.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
            )

        # Concatenate "text + vision" positions into [4, bs, seq-len]
        text_positions = text_positions[None, ...]
        position_ids = torch.cat([text_positions, vision_positions], dim=0)

        return position_ids

    def _get_image_nums_and_video_nums(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Qwen3_5Moe use timestamps and remove second_per_grid_ts
        # Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            # video_nums: (batch_size,)
            # since video_nums is the number of videos in the input dependent on the input_ids(vision_start),
            # but Qwen3_5Moe append vision_start to each frame of each video, so we need to recover the real video_nums according to video_grid_thw
            if video_grid_thw is not None:
                cumulative_frame_counts = torch.cumsum(video_grid_thw[:, 0], dim=0)
                cumulative_token_video_counts = torch.cumsum(video_nums, dim=0)
                # Find video boundaries in cumulative_frame_counts
                video_boundary_indices = torch.searchsorted(cumulative_frame_counts, cumulative_token_video_counts)
                # example: video_boundary_indices = [3, 5] means video_nums = [4, 2]
                video_nums = torch.diff(torch.cat([-video_boundary_indices.new_ones(1), video_boundary_indices]))

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if key == "position_ids" and dict_to_expand[key].ndim == 3:
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=1)
                elif (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = [
    "Qwen3_5MoeVisionModel",
    "Qwen3_5MoeTextModel",
    "Qwen3_5MoeModel",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoePreTrainedModel",
]
