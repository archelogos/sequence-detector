// @Sergio_Gordillo.
'use strict';
(function() {

	function Step(sName, sBehavior){

		if (typeof sName !== 'string' || typeof sBehavior !== 'function'){
			throw Error('Wrong params to create a node');
		}
		this.name = sName;
		var connections = [];
		var behavior = sBehavior

		var run = function(){
			behavior();
		};

		this.addConnections = function (steps){
			connections = steps;
		};

		this.move = function(stepName){
			for(var i=0; i < connections.length; i++){
				if (stepName === connections[i].name){
					run();
					return step;
				}
				else{
					throw Error('step not found or movement not allowed');
				}
			}
		};

		this.setInitial = function(){
			run();
		};

	}

	function Machine(){

		this.steps = [];
		this.current;

		this.isRegistered = function(stepName){
			for(var i=0; i < this.steps.length; i++){
				if (stepName === this.steps[i].name){
					return this.steps[i];
				}
				else{
					return false;
				}
			}
		};

		this.addSteps = function(steps){
			this.steps = steps;
		};

		this.move = function(followingStepName){
			var aux = this.current;
			this.current = aux.move(followingStepName);
		};

		this.init = function(initStepName){
			var step = this.isRegistered(initStepName);
			if(step){
				this.current = step;
				step.setInitial();
			}
			else{
				throw Error('Step '+initStepName+' doesnt exist');
			}
		};

	}

	const IMG_FOLDER = 'static/data';
	const NUM_IMG = 20;
	const NUM_SAMPLES = 200;
	const imageContainer = document.getElementById('image-container');

	let imageId;

	function getImageSample(index){
		let snippet = "<img id=\"image-sample-"+index+"\" src=\"static/data/"+index+".png\">";
		return snippet;
	}

	function predict(){
		if(typeof imageId === "undefined"){
			alert('Please, select an image!')
		}
		else{
			try{
				var xmlhttp = new XMLHttpRequest();

				xmlhttp.onreadystatechange = function() {
		        if (xmlhttp.readyState == XMLHttpRequest.DONE ) {
		           if (xmlhttp.status == 200) {
		               console.log(xmlhttp.responseText);
									 window.prediction = xmlhttp.responseText;
		           }
		           else if (xmlhttp.status == 400) {
		              alert('There was an error 400');
		           }
		           else {
		               alert('something else other than 200 was returned');
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

	// Steps Functionality

	// INIT
	function init(){
		// Show random images
		let offset = Math.round(Math.random() * (NUM_SAMPLES-NUM_IMG));
		for(var i=0; i < NUM_IMG; i++){ // images name starts at 1
			imageContainer.innerHTML += getImageSample(offset+i);
		}
		// Assign behavior
		var images = imageContainer.getElementsByTagName('img');
		[].forEach.call(images, function(image){
			image.addEventListener('click', function(e){
				var id = this.id;
				imageId = id.split("-")[id.split("-").length-1];

			});
		});
		// Assign behavior
		var button = document.getElementById('predict-button');
		button.addEventListener('click', function(e){
			predict();
		});
	};

	// SELECTED
	function selected(){

	};

	// LOADING
	function loading(){

	};

	// SUCCESS
	function success(){

	};

	// ERROR
	function error(){

	};

	// Creating STEPS
	const sINIT = new Step('INIT', init);
	const sSELECTED = new Step('SELECTED', selected);
	const sLOADING = new Step('LOADING', loading);
	const sSUCCESS = new Step('SUCCESS', success);
	const sERROR = new Step('ERROR', error);

	// Connections
	sINIT.addConnections([sSELECTED]);
	sSELECTED.addConnections([sINIT, sLOADING]);
	sLOADING.addConnections([sSUCCESS, sERROR]);
	sSUCCESS.addConnections([sINIT]);
	sERROR.addConnections([sINIT]);

	// Create Machine, add steps and initialize it
	const pMachine = new Machine();
	pMachine.addSteps([sINIT, sSELECTED, sLOADING, sSUCCESS, sERROR]);
	pMachine.init('INIT');



})(); // End of use strict
