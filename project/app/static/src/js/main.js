// @Sergio_Gordillo.
'use strict';
(function() {

	function State(sName, sBehavior){

		if (typeof sName !== 'string' || typeof sBehavior !== 'function'){
			throw Error('Wrong params to create a node');
		}
		this.name = sName;
		var connections = [];
		var behavior = sBehavior

		this.addConnections = function (states){
			connections = states;
		};

		this.move = function(stateName){
			for(var i=0; i < connections.length; i++){
				if (stateName === connections[i].name){
					return connections[i];
				}
			}
			throw Error('State not found or movement not allowed');
		};

		this.run = function(data){
			behavior(data);
		};

	}

	function Machine(){

		this.states = [];
		this.current;

		this.isRegistered = function(stateName){
			for(var i=0; i < this.states.length; i++){
				if (stateName === this.states[i].name){
					return this.states[i];
				}
			}
			return false;
		};

		this.addStates = function(states){
			this.states = states;
		};

		this.move = function(followingStateName, commonData){
			var aux = this.current;
			this.current = aux.move(followingStateName);
			this.current.run(commonData);
		};

		this.init = function(initStateName){
			var state = this.isRegistered(initStateName);
			if(state){
				this.current = state;
				state.run();
			}
			else{
				throw Error('State '+initStateName+' doesnt exist');
			}
		};

	}

	const IMG_FOLDER = 'static/data';
	const NUM_IMG = 20;
	const NUM_SAMPLES = 200;
	const N = 5;

	let imageId;

	function getImageSample(index){
		let snippet = "<img id=\"image-sample-"+index+"\" src=\"static/data/"+index+".png\">";
		return snippet;
	}

	function loadImages(imageContainer){
		let offset = Math.round(Math.random() * (NUM_SAMPLES-NUM_IMG));
		for(var i=0; i < NUM_IMG; i++){ // images name starts at 1
			imageContainer.innerHTML += getImageSample(offset+i);
		}
		var images = imageContainer.getElementsByTagName('img');
		[].forEach.call(images, function(image){
			image.addEventListener('click', function(e){
				e.preventDefault();
				var id = this.id;
				imageId = id.split("-")[id.split("-").length-1];
				var snackbarContainer = document.querySelector('#snackbar-info');
				var data = {message: 'Image Selected'};
				snackbarContainer.MaterialSnackbar.showSnackbar(data);
			});
		});
	}

	function predict(){
		if(typeof imageId === "undefined"){
			alert('Please, select an image!')
		}
		else{
			pMachine.move('LOADING');
			try{
				var xmlhttp = new XMLHttpRequest();

				xmlhttp.onreadystatechange = function() {
		        if(xmlhttp.readyState == XMLHttpRequest.DONE){
							var data = xmlhttp.responseText;
		          	if(xmlhttp.status == 200){
									pMachine.move('SUCCESS', data);
		          	}
		          	else{
									pMachine.move('ERROR', data);
		          	}
		        }
		    };
				xmlhttp.open("POST", "/api/predict", true);
				xmlhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
				xmlhttp.send(JSON.stringify({"imageId":imageId}));
			}
			catch(e){
				// Manage Error
			}
		}
	}

	function getLabels(labels){
		var L = labels[0];
		var str = "";
		for(var i = 0; i < L; i++){
			str += labels[i+1];
		}
		return str;
	}

	function showProbsList(raw_preds, probsContainer){
		var numRows = raw_preds.length/(N+1);
		for(var i=0; i < numRows; i++){
			var row = probsContainer.querySelector('#row-'+i);
			if(i === numRows - 1){ // last row
				row.innerHTML = "<td class=\"mdl-data-table__cell\">No digit</td>";
			}
			else{
				row.innerHTML = "<td class=\"mdl-data-table__cell\">"+i+"</td>";
			}
			var snippet = "";
				for(var j=i; j < raw_preds.length; j+=numRows){
					snippet += "<td class=\"mdl-data-table__cell\">"+raw_preds[j].toFixed(5)+"</td>";
				}
			row.innerHTML += snippet;
		}
	}

	function showReport(data){
		try{
			var raw_preds = data.raw_preds;
			var y = data.y;
			var labels = data.labels;
		}
		catch(e){}
		let labelContainer = document.getElementById('label-container-success');
		let predContainer = document.getElementById('prediction-container-success');
		let probsContainer = document.getElementById('probs-container-success');

		let labelsSnippet = "<div id=\"success-labels\" class=\"cell-results\"> Real Number: "+getLabels(labels)+"</div>";
		labelContainer.innerHTML = labelsSnippet;

		if(getLabels(labels) === getLabels(y)){
			var predictedSnippet = "<div id=\"success-predicted\" class=\"cell-results exact-result\"> Prediction: "+getLabels(y)+"</div>";
		}
		else{
			var predictedSnippet = "<div id=\"success-predicted\" class=\"cell-results error-result\"> Prediction: "+getLabels(y)+"</div>";
		}
		predContainer.innerHTML = predictedSnippet;

		showProbsList(raw_preds, probsContainer);

	}

	// states Functionality

	// INIT
	function init(){
		document.getElementById('state-selected').setAttribute('style', 'display:none;');
		document.getElementById('state-success').setAttribute('style', 'display:none;');
		document.getElementById('state-error').setAttribute('style', 'display:none;');
		document.getElementById('state-init').setAttribute('style', 'display:block;');
		// Load images and Functionality
		let imageContainer = document.getElementById('image-container-init');
		imageContainer.innerHTML = '';
		loadImages(imageContainer); // Initial Load
	};

	// SELECTED
	function selected(){
		document.getElementById('state-init').setAttribute('style', 'display:none;');
		document.getElementById('state-selected').setAttribute('style', 'display:block;');
		let imageContainer = document.getElementById('image-container-selected');
		imageContainer.innerHTML = getImageSample(imageId);
	};

	// LOADING
	function loading(){
		document.getElementById('state-loading').setAttribute('style', 'display:block;');
	};

	// SUCCESS
	function success(data){
		document.getElementById('state-loading').setAttribute('style', 'display:none;');
		document.getElementById('state-selected').setAttribute('style', 'display:none;');
		document.getElementById('state-success').setAttribute('style', 'display:block;');
		let imageContainer = document.getElementById('image-container-success');
		imageContainer.innerHTML = getImageSample(imageId);
		showReport(JSON.parse(data));
	};

	// ERROR
	function error(data){
		document.getElementById('state-loading').setAttribute('style', 'display:none;');
		document.getElementById('state-selected').setAttribute('style', 'display:none;');
		document.getElementById('state-error').setAttribute('style', 'display:block;');


	};

	// Buttons
	(function(){
		// Buttons INIT
		var prevButton = document.getElementById('init-button-prev');
		prevButton.addEventListener('click', function(e){
			e.preventDefault();
			let imageContainer = document.getElementById('image-container-init');
			imageContainer.innerHTML = '';
			loadImages(imageContainer);
		});
		var nextButton = document.getElementById('init-button-next');
		nextButton.addEventListener('click', function(e){
			e.preventDefault();
			if(typeof imageId === "undefined"){
				alert('Please, select an image')
			}
			else{
				pMachine.move('SELECTED');
			}
		});

		// Buttons SELECTED
		var prevButton = document.getElementById('selected-button-prev');
		prevButton.addEventListener('click', function(e){
			e.preventDefault();
			pMachine.move('INIT');
		});
		var nextButton = document.getElementById('selected-button-next');
		nextButton.addEventListener('click', function(e){
			e.preventDefault();
			predict();
		});

		// Buttons SUCCESS
		var nextButton = document.getElementById('success-button-next');
		nextButton.addEventListener('click', function(e){
			e.preventDefault();
			pMachine.move('INIT');
		});

		// Buttons ERROR
		var nextButton = document.getElementById('error-button-next');
		nextButton.addEventListener('click', function(e){
			e.preventDefault();
			pMachine.move('INIT');
		});

	})();

	// Creating States
	const sINIT = new State('INIT', init);
	const sSELECTED = new State('SELECTED', selected);
	const sLOADING = new State('LOADING', loading);
	const sSUCCESS = new State('SUCCESS', success);
	const sERROR = new State('ERROR', error);

	// Connections
	sINIT.addConnections([sSELECTED]);
	sSELECTED.addConnections([sINIT, sLOADING]);
	sLOADING.addConnections([sSUCCESS, sERROR]);
	sSUCCESS.addConnections([sINIT]);
	sERROR.addConnections([sINIT]);

	// Create Machine, add states and initialize it
	const pMachine = new Machine();
	pMachine.addStates([sINIT, sSELECTED, sLOADING, sSUCCESS, sERROR]);
	pMachine.init('INIT');



})(); // End of use strict
