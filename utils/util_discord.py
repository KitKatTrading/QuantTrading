from discord_webhook import DiscordWebhook, DiscordEmbed

def send_image_to_discord(
        path_image,   # path to the image to be sent
        title_message,  # title_message to send
        webhook,  # HTTP webhook link
):
    # create embed object for webhook
    embed_Chinese = DiscordEmbed(title=title_message, color='03b2f8')
    # set local attachment for image
    with open(path_image, "rb") as f:
        filename_random = ''.join(random.choices(string.ascii_lowercase, k=10)) + '.jpeg'
        webhook.add_file(file=f.read(), filename=filename_random)
    embed_Chinese.set_image(url='attachment://' + filename_random)
    webhook.add_embed(embed_Chinese)
    response = webhook.execute()
    webhook.remove_files()
    webhook.remove_embeds()
    f.close()