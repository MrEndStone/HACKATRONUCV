export const ajaxPost = async (url, values) => {
    const uri = url;
    const form = new FormData();
    for (const property in values) {
        form.append(property, values[property])
    }
    const data = await fetch(uri, {
        method: 'POST',
        body: form
    })
    .then(res => res.json())
    .catch(err => err);
    return data;
}

export const ajaxGet = async (url) => {
    const uri = url;
    const data = await fetch(uri, {
        method: 'GET'
    })
    .then(res => res.json())
    .catch(err => err);
    return data;
}