var stocks = []

function addStock(stock){
    currentStock = document.getElementById("stock").value
    stocks.push(currentStock)
    console.log(stocks)
    var form = document.getElementById("myform")
    form.reset()
}